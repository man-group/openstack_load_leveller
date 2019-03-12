#!/usr/bin/env python
# coding: utf-8
""" Openstack load leveller script

This script attempts to "load balance" openstack, by live-migrating instances
from busy hypervisors to less busy hypervisors.

The busyness is calculated from the memory and cpu utilization (as reported by
prometheus).

The openstack credentials etc are stored in /etc/loadleveller-secrets.conf (by
default).

Exit codes:

* EX_SOFTWARE (70) if Prometheus can't be contacted, VMs are lost etc.
* EX_IOERR (74) for IO errors, specifically when reading in the config.
* EX_NOPERM (77) If openstack denies an action based on permissions.
* EX_CONFIG (78) For config errors, including requesting openstack resources
    that do not exist.
"""

import json
import logging
import operator
import os
import pprint
import random
import sys
import time
import urllib2

import attr
import dotenv
import enum
import urllib3

# Silence SSL cert errors so that the Openstack client libs don't spam us to death with warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import click
import keystoneauth1
import novaclient
from novaclient import client

DEFAULT_OS_AUTH_URL = "https://openstackapi:5000/v3/"
DEFAULT_OS_LIMIT_TO_AZ = "nova"
DEFAULT_OS_LIMIT_TO_EXACT_CPU_SPEC_MATCH = "FALSE"

DEFAULT_PROMETHEUS_CPU_USED = 'sort_desc(1 - avg_over_time(avg:cpu_usage_idle{openstack_role="compute", region="gdc"}[5m])/100)'
DEFAULT_PROMETHEUS_MEM_USED = 'sort_desc(avg_over_time(mem_used_percent{openstack_role="compute", region="gdc"}[1m])/100)'
DEFAULT_PROMETHEUS_QUERY_URL = "http://prometheus:9090/api/v1/query"

# Relative weights of how important the metrics are for the synthetic scoring.
DEFAULT_MEM_WEIGHT = 2.0
DEFAULT_CPU_WEIGHT = 0.4

# If the most loaded hypervisor has a syn score less than this, don't do anything
DEFAULT_MIN_LOAD_BEFORE_MOVE = 1.5

# If the most loaded hypervisor is less than x% more loaded than the least loaded hypervisor, do nothing
# Note. the metric is scaled to 0-1
DEFAULT_MIN_LOAD_DIFF_BEFORE_MOVE = 0.1

# Openstack API version to use
DEFAULT_API_VERSION = "2.29"

# How long do we sleep between a migration finishing, and a new one starting
DELAY_BETWEEN_MIGRATIONS = 10  # seconds

# How long do we sleep between checking the status of a current migration
DELAY_BETWEEN_STATUS_CHECK = 10  # seconds


class MigrationStatus(enum.Enum):
    to_be_migrated = 1
    migrating = 2
    done = 3
    timed_out = 4
    failed = 5
    delayed = 6
    queued = 7
    invalid = 8
    vm_not_active = 9
    vm_didnt_move = 10
    vm_lost = 11


@attr.s(frozen=True)
class HostMigration(object):
    name = attr.ib()
    uuid = attr.ib()
    destination = attr.ib()
    source_hypervisor = attr.ib()
    status = attr.ib()  # MigrationStatus
    percent_complete = attr.ib(default=0, type=int)


@attr.s(frozen=True)
class LoadbalancerConfig(object):
    """ Class to hold openstack specific bits of config """

    # Openstack credentials
    mem_metric_weight = attr.ib(default=DEFAULT_MEM_WEIGHT)
    cpu_metric_weight = attr.ib(default=DEFAULT_CPU_WEIGHT)
    min_load_before_move = attr.ib(default=DEFAULT_MIN_LOAD_BEFORE_MOVE)
    min_load_diff_before_move = attr.ib(default=DEFAULT_MIN_LOAD_DIFF_BEFORE_MOVE)
    username = attr.ib(default=None)
    password = attr.ib(default=None)
    domain = attr.ib(default=None)
    project_id = attr.ib(default=None)
    auth_url = attr.ib(default=None)
    auth_file = attr.ib(default="/etc/loadleveller-secrets.conf")
    api_version = attr.ib(default=DEFAULT_API_VERSION)
    insecure = attr.ib(default=False)

    limit_to_az = attr.ib(default=None)
    limit_to_exact_cpu_spec = attr.ib(default=True)

    # Prometheus queries etc
    prometheus_query_url = attr.ib(
        default="http://the_prometheus_server:9090/api/v1/query"
    )
    prometheus_query_mem_used = attr.ib(default=None)
    prometheus_query_cpu_used = attr.ib(default=None)

    # class method to load the config and stuff it into the ,conf variable
    @classmethod
    def load_config(cls, filename):
        # type: (LoadbalancerConfig, str) -> LoadbalancerConfig
        """ Read the config file and parse it.

        raises:
            ConfigParser.NoOptionError      (missing option in file)
            ConfigParser.NoSectionError     (missing config file (!!), no perms to config file, or missing section in file)
        """
        dotenv.load_dotenv(dotenv_path=filename)

        is_insecure = os.getenv("OS_INSECURE", "FALSE").upper() in ["TRUE", "YES"]

        mem_metric_weight = float(
            os.getenv("PROMETHEUS_METRIC_WEIGHT_MEM", DEFAULT_MEM_WEIGHT)
        )
        cpu_metric_weight = float(
            os.getenv("PROMETHEUS_METRIC_WEIGHT_CPU", DEFAULT_CPU_WEIGHT)
        )
        min_load_before_move = float(
            os.getenv(
                "PROMETHEUS_MIN_LOAD_BEFORE_MOVE", DEFAULT_MIN_LOAD_BEFORE_MOVE
            )
        )
        min_load_diff_before_move = float(
            os.getenv(
                "PROMETHEUS_MIN_LOAD_DIFF_BEFORE_MOVE",
                DEFAULT_MIN_LOAD_DIFF_BEFORE_MOVE,
            )
        )

        limit_to_exact_cpu_spec = os.getenv(
            "OS_LIMIT_TO_EXACT_CPU_SPEC_MATCH", DEFAULT_OS_LIMIT_TO_EXACT_CPU_SPEC_MATCH
        ).upper() in ["TRUE", "YES"]

        api_version = os.getenv("OS_API_VERSION", DEFAULT_API_VERSION)
        auth_url = os.getenv("OS_AUTH_URL", DEFAULT_OS_AUTH_URL)
        project_id = os.getenv("OS_PROJECT_ID")
        limit_to_az = os.getenv("OS_LIMIT_TO_AZ", DEFAULT_OS_LIMIT_TO_AZ)
        prometheus_query_url = os.getenv("PROMETHEUS_QUERY_URL", DEFAULT_PROMETHEUS_QUERY_URL)
        prometheus_query_cpu_used=os.getenv("PROMETHEUS_CPU_USED", DEFAULT_PROMETHEUS_CPU_USED)

        try:
            return LoadbalancerConfig(
                # Required settings - FIXME:  read these in and warn in a better way than an exception.
                username=os.environ["OS_USERNAME"],
                domain=os.environ["OS_DOMAIN_NAME"],
                password=os.environ["OS_PASSWORD"],
                prometheus_query_mem_used=DEFAULT_PROMETHEUS_MEM_USED,
                prometheus_query_cpu_used=DEFAULT_PROMETHEUS_CPU_USED,
                # Optional settings with default values
                prometheus_query_url=prometheus_query_url,
                auth_url=auth_url,
                project_id=project_id,
                api_version=api_version,
                insecure=is_insecure,
                mem_metric_weight=mem_metric_weight,
                cpu_metric_weight=cpu_metric_weight,
                min_load_before_move=min_load_before_move,
                min_load_diff_before_move=min_load_diff_before_move,
                limit_to_az=limit_to_az,
                limit_to_exact_cpu_spec=limit_to_exact_cpu_spec,
            )
        except KeyError as e:
            raise ConfigError("Missing required config option {}".format(e))


class HypervisorNotFoundException(IndexError):
    pass


class VMNotFound(Exception):
    pass


class VMNotActiveStatus(EnvironmentError):
    pass


class FlavourNotFound(EnvironmentError):
    pass


class NoHypervisorsFound(EnvironmentError):
    pass


class AvailabilityZoneNotFound(EnvironmentError):
    pass


class PrometheusError(EnvironmentError):
    pass


class ConfigError(EnvironmentError):
    pass


# Function to do a http query and return a dict
def do_query(query_url, query):
    # type: (str, str) -> str
    """ Run a query against prometheus and return the result as a dict

    raises:
        PrometheusError
    """

    try:
        url = "{}?query={}".format(query_url, urllib2.quote(query, safe='/:"'))
        f = urllib2.urlopen(url)
    except urllib2.URLError as e:
        raise PrometheusError(
            "Couldn't connect to prometheus URL {}  (Exception text: {})".format(url, e)
        )

    try:
        return json.loads(f.read())
    except ValueError as e:
        raise PrometheusError(
            "Prometheus URL {} couldn't be loaded by the JSON library "
            "(Exception text: {})".format(url, e)
        )


def get_metrics(query_url, query):
    # type: (str, str) -> dict(dict)
    """ Get all metrics for the type, and return as:
        { 'host': { instance: inst,  value: val } }
    raises:
        PrometheusError
    """
    query_result = do_query(query_url, query)
    results = {}

    try:
        for r in query_result["data"]["result"]:
            host = r["metric"]["host"]
            instance = r["metric"]["instance"]
            value = r["value"][1]
            results[host] = {"instance": instance, "value": float(value)}
    except (IndexError, KeyError, TypeError) as e:
        raise PrometheusError(
            "Prometheus returned JSON results without the requisite keys "
            "(Exception text: {})".format(e)
        )
    return results


def calculate_one_score(conf, host, cpu_metrics, mem_metrics):
    # type: (LoadbalancerConfig, str, dict, dict) -> dict
    """ Calculate the synthetic score for one box.
        Returns a dict with host, syn_score, cpu_score, mem_score """

    cpu_score = cpu_metrics[host]["value"] * conf.cpu_metric_weight
    mem_score = mem_metrics[host]["value"] * conf.mem_metric_weight

    return {
        "host": host,
        "syn_score": cpu_score + mem_score,
        "cpu_score": cpu_score,
        "mem_score": mem_score,
    }


def calculate_synthetic_scores(conf):
    # type: LoadbalancerConfig -> list(dict)
    """ Calculate the synthetic load score for each box (that we care about - filter on availability zone
        if appropriate).
        Returns list of dicts with keys: host, syn_score, cpu_score, mem_score
    raises:
        PrometheusError
    """

    cpu_metrics = get_metrics(conf.prometheus_query_url, conf.prometheus_query_cpu_used)
    mem_metrics = get_metrics(conf.prometheus_query_url, conf.prometheus_query_mem_used)

    # First, compute a list of hosts
    all_hosts = set(cpu_metrics.keys() + mem_metrics.keys())

    # Calculate a list of host score dicts
    host_score_list = [
        calculate_one_score(conf, k, cpu_metrics, mem_metrics) for k in all_hosts
    ]

    return sorted(host_score_list, key=operator.itemgetter("syn_score"), reverse=True)


def get_openstack_connection_object(conf):
    # type: LoadbalancerConfig -> OpenstackClient
    """ Create openstack novaclient object (with auth details etc).
    raises:
        Nothing - this just creates a client obj (and doesn't e.g. authenticate)
    """

    return novaclient.client.Client(
        conf.api_version,
        conf.username,
        conf.password,
        conf.project_id,
        conf.auth_url,
        insecure=conf.insecure,
        user_domain_name=conf.domain,
    )


def find_servers_on_host_with_state(nova, name, state="ACTIVE"):
    # type: (novaclient.client.Client, str, str) -> list[Openstack.Server]
    """ Return the list of VMs on a hypervisor with name 'name' in state state.

    raises:
        HypervisorNotFoundException
        VMNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    # The hypervisor_hostname is _not_ the same as the ['service']['host'] field is, so
    # we find the hypervisor object, to find the actual hypervisor hostname, so that we
    # can use that in the nova.hypervisors.search and get the VMs on the hypervisor.
    hypervisor_obj = find_hypervisor_from_id(nova, name)
    hypervisor_hostname = hypervisor_obj.to_dict()["hypervisor_hostname"]

    try:
        hypervisor_with_servers = nova.hypervisors.search(
            hypervisor_hostname, servers=True
        )
    except novaclient.exceptions.NotFound:
        # E.g. hypervisor_hostname is an invalid name
        raise HypervisorNotFoundException(
            "Can't find hypervisor with id {}".format(name)
        )

    if not hypervisor_with_servers:
        raise HypervisorNotFoundException(
            "Can't find hypervisor with id {}".format(name)
        )

    # servers here is a list of dicts of the form:
    #    {u'uuid': u'331e22d6-....-9378f98', u'name': u'instance-0000nnnn'}
    servers = hypervisor_with_servers[0].to_dict()["servers"]

    # We need to call get_vm_details on each 'server' here to find out their actual state
    return [
        s
        for s in servers
        if get_vm_details(nova, s["uuid"]).to_dict()["status"] == state
    ]


def get_vm_details(nova, uuid):
    # type: (OpenstackClient, str) -> Openstack.Server
    """ Find the (vm) server object with a particular uuid.

    raises:
        VMNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    try:
        vm = nova.servers.get(uuid)
    except novaclient.exceptions.NotFound:
        raise VMNotFound("Can't find vm with uuid {}".format(uuid))

    if vm is None:
        raise VMNotFound(
            "Can't find vm with uuid {}, but openstack API did not throw error".format(
                uuid
            )
        )
    return vm


def get_migration_completion_percentage(nova_status):
    # type: dict -> float
    """ Return how complete the migration is (in percent)"""
    if (
        "memory_total_bytes" in nova_status
        and nova_status["memory_total_bytes"] is not None
    ):
        mem_total_bytes = int(nova_status.get("memory_total_bytes", 0)) or 0
        mem_remaining_bytes = int(nova_status.get("memory_remaining_bytes", 0))

        if mem_total_bytes == 0:
            # This happens at the very beginning of a migration
            return 0.0
        else:
            return 100.0 * (1.0 - float(mem_remaining_bytes) / float(mem_total_bytes))
    return 0.0


def update_migration_status(nova, vm_migration):
    # type: (OpenstackClient, MigrationStatus) -> MigrationStatus
    """ Update the migration status for a server.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    if vm_migration.status == MigrationStatus.done:
        logging.debug("VM {} migrated, we're done.".format(vm_migration))
        return vm_migration

    logging.debug("Getting migration status for uuid {}".format(vm_migration.uuid))

    def _find_vm(vm_migration, server_migrations):
        for m in server_migrations:
            if m.to_dict()["server_uuid"] == vm_migration.uuid:
                # This is "our" migration
                logging.debug("{}".format(pprint.pformat(m.to_dict())))
                return m.to_dict()

    server_migrations = nova.server_migrations.list(vm_migration.uuid)

    nova_status = _find_vm(vm_migration, server_migrations)
    if nova_status:
        return attr.evolve(
            vm_migration,
            status=MigrationStatus.migrating,
            percent_complete=get_migration_completion_percentage(nova_status),
        )

    # When we are state migrating, but Nova no longer has a state for it,
    # the migration is complete and we can return done.
    if vm_migration.status == MigrationStatus.migrating:
        return attr.evolve(
            vm_migration, status=MigrationStatus.done, percent_complete=100
        )

    # Still waiting, no need to update anything.
    return vm_migration


def vm_live_migration(nova, vm_migration):
    # type: (OpenstackClient, MigrationStatus) -> MigrationStatus
    """ Start the move a VM elsewhere. The uuid parameter is the UUID of a VM.
        Optionally dest can be set to a hypervisor hostname to move to.

        Note that this function does not wait for the migration to finish,
        it merely starts it.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
        novaclient.exceptions.BadRequest     (e.g. incompatible CPU spec)

    """
    try:
        vm = get_vm_details(nova, vm_migration.uuid).to_dict()
    except VMNotFound:
        return attr.evolve(vm_migration, status=MigrationStatus.invalid)

    logging.info(
        "Attempting to move VM {} from hypervisor {} to {}".format(
            vm["name"], vm_migration.source_hypervisor, vm_migration.destination
        )
    )
    logging.debug("VM details:\n{}".format(pprint.pformat(vm)))

    # Make sure the VM is actually in the active status, otherwise migration
    # will fail.
    if vm["status"] == "MIGRATING":
        # Already migrating
        logging.warn(
            "VM {} (uuid {}) is already migrating".format(vm["name"], vm_migration.uuid)
        )
        # return HostMigration(uuid, MigrationStatus.migrating)
        return attr.evolve(vm_migration, status=MigrationStatus.migrating)
    elif vm["status"] != "ACTIVE":
        # I.e. it's either ERROR or MIGRATING or some other status...
        return attr.evolve(vm_migration, status=MigrationStatus.invalid)
    else:
        dest = vm_migration.destination
        nova.servers.live_migrate(vm["id"], dest, "auto")

    return attr.evolve(vm_migration, status=MigrationStatus.migrating)


def select_vm_to_move_off_host(nova, hypervisor):
    # type: (OpenstackClient, str) ->  dict
    """ Find a list of VMs on the hypervisor host, and select one to be moved off

    raises:
        HypervisorNotFoundException
        VMNotFound                      (from get_vm_details)
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    # Get list of VMs on the host
    servers = find_servers_on_host_with_state(nova, hypervisor, "ACTIVE")

    if not servers:
        return None
    return random.choice(servers)


def verify_move_happened_successfully(nova, vm_migration):
    # type: (OpenstackClient, MigrationStatus) -> MigrationStatus
    """ Make sure the move happened successfully, i.e. we're not in
        ERROR state, and that we're not still on the original hypervisor.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    try:
        vm = get_vm_details(nova, vm_migration.uuid).to_dict()
    except VMNotFound:
        # The VM has disappeared!
        return attr.evolve(vm_migration, status=MigrationStatus.vm_lost)

    # Check that everything is well with the VM and the move
    if vm["status"] != "ACTIVE":
        # VM is not active any more - this is probably bad!
        vm_migration = attr.evolve(vm_migration, status=MigrationStatus.vm_not_active)
    elif vm["OS-EXT-SRV-ATTR:host"] == vm_migration.source_hypervisor:
        # VM is still on the original hypervisor
        vm_migration = attr.evolve(vm_migration, status=MigrationStatus.vm_didnt_move)

    return vm_migration


def setup_logging(debug=False):
    # type: Boolean -> logging.Logger
    """ simple debug logging to a file

    raises:
        IOError
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        filename="openstack-lb-debug.log",
    )

    # Make dumps to screen less verbose
    console = logging.StreamHandler()
    if debug:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def find_flavour_from_id(nova, uuid):
    # type: (OpenstackClient, str) -> Openstack.Flavour
    """ Find the flavour with specific uuid

    raises:
        FlavourNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    flavours = find_flavours(nova)

    matching_flavours = filter(lambda x: x.to_dict()["id"] == uuid, flavours)

    if not matching_flavours:
        raise FlavourNotFound("Can't find matching flavour for id {}".format(uuid))

    return matching_flavours[0]


def max_vm_migration_time(nova, vm_migration):
    # type: (OpenstackClient, str) -> int
    """ Determine the maximum migration time of a VM before declaring
    that the migration is either slow (and maybe force completing).

    raises:
        VMNotFound
        FlavourNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    # we seem to take about 240 seconds for a 64Gb machine, so about 250Mb / s
    transfer_rate = 64000.0 / 250.0
    # Allow a VM to be 50% slower in transfer before declaring it slow
    grace_factor = 1.5

    vm = get_vm_details(nova, vm_migration.uuid)
    try:
        flavour_id = vm.to_dict()["flavor"]["id"]
        flavour = find_flavour_from_id(nova, flavour_id)
    except (FlavourNotFound, KeyError, AttributeError):
        # Looks like we have incomplete info in the dict - this would indicate some openstack API issue,
        # so we raise here
        raise FlavourNotFound(
            "Flavour not found for VM {}/{}".format(
                vm_migration.name, vm_migration.uuid
            )
        )

    # the size of the VM (or rather its flavour)
    size = flavour.to_dict()["ram"]
    return (size / transfer_rate) * grace_factor


def find_flavours(nova):
    # type: (OpenstackClient) -> Openstack.Flavor
    """ Return all flavours (as list of Flavor.
    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    return nova.flavors.list()


def find_flavour_of_vm(nova, vm):
    # type: (OpenstackClient, dict) -> str
    """ Return the flavour of a particular VM.

    raises:
        VMNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    return get_vm_details(nova, vm["uuid"]).to_dict()["flavor"]["id"]


def find_hypervisor_from_id(nova, id):
    # type: (OpenstackClient, str) -> Openstack.Hypervisor
    """ Find and return a hypervisor object.
    If no hypervisor can be found matching the id, then return None.

    raises:
        HypervisorNotFoundException
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    hypervisor_list = nova.hypervisors.list(detailed=True)

    # Find all hypervisors matching the id
    matching_hvs = [
        hv for hv in hypervisor_list if hv.to_dict()["service"]["host"] == id
    ]

    # If we have any, return the first one, otherwise return None.
    # There should only ever be one, of course, but the API gives us a list.
    if not matching_hvs:
        raise HypervisorNotFoundException("Can't find hypervisor with id {}".format(id))

    return matching_hvs[0]


def find_hypervisors_that_can_fit_vm(conf, nova, vm, syn_scores):
    # type: (LoadbalancerConfig, OpenstackClient, dict, list[dict]) ->  list
    """ Find all hypervisors that could fit a vm of size vm_size.

    raises:
        VMNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    vm_flavour = find_flavour_of_vm(nova, vm)

    selected_flavour = find_flavour_from_id(nova, vm_flavour)
    flavour_ram = selected_flavour.to_dict()["ram"]

    # Build a list of hypervisors, so that we can see how much RAM they have, and make it a dict
    # for our filter function below.
    hypervisor_list = nova.hypervisors.list(detailed=True)

    hv_ram = {}
    for h in hypervisor_list:
        hd = h.to_dict()
        hv_ram[hd["service"]["host"]] = hd["memory_mb"]
        logging.debug(
            "Hypervisor total memory:   host {} -> {} Mb".format(
                hd["service"]["host"], hd["memory_mb"]
            )
        )

    # Note, this is a list of hypervisors that can fit the VM in syn_score format [ host, syn score, cpu, mem ]
    # Also, refuse to fit onto something that would be > 0.95 used
    hvs_that_fit = filter(
        lambda x: x["mem_score"] / conf.mem_metric_weight
        + float(flavour_ram) / float(hv_ram[x["host"]])
        <= 0.95,
        syn_scores,
    )

    return hvs_that_fit


def find_az_from_name(nova, az_name):
    # type: (OpenstackClient, str) -> Openstack.AvailabilityZone
    """ Return the availability zone called az_name.
        If no AZ with the name az_name exists, return None.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    azs = nova.availability_zones.list(detailed=True)

    # Sadly we have to iterate across here
    for az in azs:
        if az_name == az.to_dict()["zoneName"]:
            return az
    return None


def find_az_from_hypervisor_name(nova, hv_name):
    # type: (OpenstackClient, str) -> Openstack.AvailabilityZone
    """ Returns the availability zone object for a hypervisor with name hv_name.

        If the hypervisor cannot be found, returns None. If the openstack user does not
        have the rights to list hypervisors, we raise nova_exceptions.Forbidden.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    azs = nova.availability_zones.list(detailed=True)

    # Sadly we have to iterate across here
    for az in azs:
        if hv_name in az.to_dict()["hosts"]:
            return az
    return None


def find_active_hypervisors_in_az(az):
    # type: (Openstack.AvailabilityZone) -> list
    """ Return a simple list of host names that are active and available from an availability
        zone dict. """

    hosts_in_az = az.to_dict()["hosts"]

    return [
        h
        for h in hosts_in_az
        if hosts_in_az[h]["nova-compute"]["active"]
        and hosts_in_az[h]["nova-compute"]["available"]
    ]


def select_hypervisor_to_move_to(conf, nova, vm, syn_scores):
    # type: (LoadbalancerConfig, OpenstackClient, dict, list[dict]) -> str
    """ Returns name of a hypervisor to move the VM to. If no hypervisor fits, returns None.

    raises:
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    vm_details = get_vm_details(nova, vm["uuid"])
    logging.debug(
        "select_hypervisor_to_move_to for vm {} / {}".format(
            vm_details.to_dict()["name"], vm_details.to_dict()["id"]
        )
    )

    # Find all hypervisors
    hypervisor_list = nova.hypervisors.list(detailed=True)

    # transform the list of dict (actually objs) to a dict of hypervisor-name -> hypervisor objs
    hvs_by_name = {hv.to_dict()["service"]["host"]: hv for hv in hypervisor_list}

    # Find the dict for the current hypervisor (that the VM is on) from the list of dicts
    # in the hypervisor_list
    current_hv_name = vm_details.to_dict()["OS-EXT-SRV-ATTR:host"]
    current_hv = hvs_by_name[current_hv_name]
    logging.debug("current HV:\n{}".format(pprint.pformat(current_hv_name)))

    # Find the hypervisors that are not currently disabled
    non_disabled_hypervisors = [
        hv for hv in hypervisor_list if hv.to_dict()["status"] == "enabled"
    ]

    # Exclude the 'current' hypervisor
    hypervisors_excluding_current = [
        hv
        for hv in non_disabled_hypervisors
        if hv.to_dict()["id"] != current_hv.to_dict()["id"]
    ]

    # Find the list of hypervisors in the right AZ
    current_az = find_az_from_hypervisor_name(
        nova, vm_details.to_dict()["OS-EXT-SRV-ATTR:host"]
    )
    active_hypervisors_in_current_az = find_active_hypervisors_in_az(current_az)
    hypervisors_in_right_az = [
        hv
        for hv in hypervisors_excluding_current
        if hv.to_dict()["service"]["host"] in active_hypervisors_in_current_az
    ]

    # and then finally find the hypervisors that have the right CPU spec.
    if conf.limit_to_exact_cpu_spec:
        # hypervisors_matching_current_spec = [ hv for hv in hypervisors_in_right_az
        hypervisors_matching_current_spec = [
            hv
            for hv in hypervisors_in_right_az
            if hv.to_dict()["cpu_info"]
            == hvs_by_name[current_hv_name].to_dict()["cpu_info"]
        ]
    else:
        # ... consider anything (i.e. do not filter on CPU spec)
        hypervisors_matching_current_spec = hypervisors_in_right_az

    logging.debug(
        "Number of potential targets: {}".format(len(hypervisors_matching_current_spec))
    )

    # Now, determine which hypervisor to actually attempt to move to, by finding the hypervisor with the lowest
    # syn_score
    # Build up a list of 'scores' for the hardware that is enabled and appropriate (from a CPU arch point of
    # view, AZ etc, as determined above).

    # First filter the synthetic scores so that the list only contains scores for valid targets
    matching_hv_names = [
        hv.to_dict()["service"]["host"] for hv in hypervisors_matching_current_spec
    ]
    filtered_scores = [s for s in syn_scores if s["host"] in matching_hv_names]

    # Then, pick the lowest scored hypervisor from the list of hypervisors that can fit the current VM
    try:
        # Calculate where the VM would fit, and then select the lowest score one that does
        selected_hv_service_name = find_hypervisors_that_can_fit_vm(
            conf, nova, vm, filtered_scores
        )[-1]["host"]
        logging.debug(
            "Determined that we're using hypervisor {} for this VM".format(
                selected_hv_service_name
            )
        )

        return selected_hv_service_name

    except IndexError:
        # No hypervisors can fit a VM this size at the moment...
        return None


def ticker(max_wait_time, sleep_interval=DELAY_BETWEEN_STATUS_CHECK):
    start = time.time()
    while time.time() - start < max_wait_time:
        yield sleep_interval
        logging.debug("ticker: Sleeping {} seconds".format(sleep_interval))
        time.sleep(sleep_interval)


def wait_for_migration(nova, vm_migration):
    # type: (OpenstackClient, HostMigration) -> HostMigration
    """ Wait for a migration to either fail, or time out, or complete.

    raises:
        FlavourNotFound
        VMNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    logging.debug("Waiting for {} to finish migration".format(vm_migration.name))
    logging.debug("vm migration details: {}".format(pprint.pformat(vm_migration)))

    try:
        max_wait_time = max_vm_migration_time(nova, vm_migration)
    except VMNotFound:
        # Oops - looks like the VM has disappeared! we'll return this status so
        # that the main loop can handle this
        return attr.evolve(vm_migration, status=MigrationStatus.vm_lost)
    # max_vm_migration_time could also raise FlavourNotFound, but this would be an extremely weird event with
    # corrupted API responses or similar, so we let that bubble out.

    cur_status = None
    while True:
        for dummy_var in ticker(max_wait_time):
            # Get current status of migration
            vm_migration = update_migration_status(nova, vm_migration)

            if cur_status != vm_migration.status:
                logging.info("New migration status: {}".format(vm_migration.status))
                cur_status = vm_migration.status

            if vm_migration.status == MigrationStatus.migrating:
                logging.info(
                    "Migration {0:.2f}% complete.".format(vm_migration.percent_complete)
                )

            if vm_migration.status == MigrationStatus.timed_out:
                logging.error(
                    "Migration status not found for vm {}. Did "
                    "migration even start?".format(vm_migration)
                )
                return attr.evolve(vm_migration, status=MigrationStatus.timed_out)

            if vm_migration.status == MigrationStatus.done:
                return vm_migration

        # We end up here if we've spent more than max_wait_time waiting. At the moment we don't force complete, but
        # rather just dump out a warning message.
        logging.warn(
            "Migration of {} not completed within {} seconds, but will not force complete.".format(
                vm_migration, max_wait_time
            )
        )


def try_migrate_vm(nova, migrate_fn, vm_migration):
    # type: (OpenstackClient, function, HostMigration) -> HostMigration
    """ Attempt to migrate a VM by running the migrate_fn function against it.

    raises:
        FlavourNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    # start migration
    vm_migration = migrate_fn(nova, vm_migration)

    if vm_migration.status == MigrationStatus.invalid:
        # Something went wrong here - we can't find the Hypervisor or VM, for example
        return vm_migration

    vm_migration = wait_for_migration(nova, vm_migration)
    if vm_migration.status == MigrationStatus.timed_out:
        return vm_migration

    # Once migration is complete, let's make sure the box is not in ERROR,
    # and that it actually moved, wasn't lost etc
    vm_migration = verify_move_happened_successfully(nova, vm_migration)
    if vm_migration.status in [
        MigrationStatus.vm_not_active,
        MigrationStatus.vm_didnt_move,
        MigrationStatus.vm_lost,
    ]:
        return vm_migration

    return attr.evolve(vm_migration, status=MigrationStatus.done)


def dummy_migration(nova, vm_migration):
    # type: (OpenstackClient, MigrationStatus) -> MigrationStatus
    """ Dummy migration function that just returns a Migrationstatus.done immediately.
    Note that nova and dest are never used, but kept here because otherwise the signature
    changes. """
    logging.info(
        "Would migrate VM {}/{} to hypervisor {}".format(
            vm_migration.name, vm_migration.uuid, vm_migration.destination
        )
    )
    vm_migration = attr.evolve(vm_migration, status=MigrationStatus.done)
    return vm_migration


def filter_syn_scores_to_az(conf, nova, org_scores):
    # type: (LoadbalancerConfig, OpenstackClient, list[dict]) -> list[dict]
    """ Filter out hosts that are not in the correct AZ from a list of synthetic scores.

        If limit_to_az is None in the config, then return the original unfiltered scores

    raises:
        NoHypervisorsFound
        AvailabilityZoneNotFound
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """

    if conf.limit_to_az is None:
        logging.debug(
            "Not limiting hosts based on availability zone (conf.limit_to_az is None)"
        )
        return org_scores

    logging.info("Limiting valid hosts to AZ {}".format(conf.limit_to_az))

    # Find hypervisors in AZ
    az = find_az_from_name(nova, conf.limit_to_az)
    if az:
        valid_hostnames = find_active_hypervisors_in_az(az)

        if len(valid_hostnames) == 0:
            raise NoHypervisorsFound(
                "Can't find any hypervisors in availablity zone {}".format(
                    conf.limit_to_az
                )
            )
    else:
        # No valid hypervisors found in AZ - bomb out
        raise AvailabilityZoneNotFound(
            "Can't find availablity zone {}".format(conf.limit_to_az)
        )

    # Filter out hypervisors that are not in the list of valid ones
    valid_scores = [x for x in org_scores if x["host"] in valid_hostnames]

    return valid_scores


def create_vm_migrations(conf, nova):
    # type: (LoadbalancerConfig, OpenstackClient) -> collections.Iterable
    """ Generate HostMigration objects for moves

    raises:
        NoHypervisorsFound
        AvailabilityZoneNotFound
        PrometheusError
        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
    """
    while True:
        logging.info("Calculating synthetic load scores")
        syn_scores = calculate_synthetic_scores(conf)
        syn_scores = filter_syn_scores_to_az(conf, nova, syn_scores)
        logging.debug("Scores: {}".format(pprint.pformat(syn_scores)))

        # Get the most loaded server
        most_loaded = syn_scores[0]
        least_loaded = syn_scores[-1]
        load_difference_pct = most_loaded["syn_score"] / least_loaded["syn_score"] - 1.0

        if most_loaded["syn_score"] < conf.min_load_before_move:
            # No server loaded enough to bother with migrations
            logging.info("No hypervisor is loaded enough to bother migrating VMs")
        elif load_difference_pct < conf.min_load_diff_before_move:
            # Load difference is too small between hypervisors to bother moving
            logging.info(
                "Load difference between hypervisors is too low to move anything (diff: {}, limit: {})".format(
                    load_difference_pct, conf.min_load_diff_before_move
                )
            )
        else:
            logging.info(
                "The most loaded host is {} (syn_score: {}, mem_score: {}, cpu_score: {})".format(
                    most_loaded["host"],
                    most_loaded["syn_score"],
                    most_loaded["mem_score"],
                    most_loaded["cpu_score"],
                )
            )

            # Pick a VM to move
            try:
                selected_vm = select_vm_to_move_off_host(nova, most_loaded["host"])
                if selected_vm is None:
                    logging.info("Can't find any VMs on hypervisor - trying again...")
                    continue

                vm_details = get_vm_details(nova, selected_vm["uuid"]).to_dict()
            except VMNotFound:
                logging.warn(
                    "Attempted to get vm details for {}, but failed - trying other vm".format(
                        selected_vm["uuid"]
                    )
                )
                continue  # Try to find another one...
            except HypervisorNotFoundException as e:
                logging.warn(
                    "Exception whilst trying to find a hypervisor. Details: {}".format(
                        e
                    )
                )
                continue  # Try again

            # Select a hypervisor to move it to. Returns None if no hypervisor fits the workload
            selected_dest = select_hypervisor_to_move_to(
                conf, nova, selected_vm, syn_scores
            )

            if selected_dest is not None:

                # Print info about our target
                host_to_score = {s["host"]: s for s in syn_scores}
                logging.info(
                    "Moving to hypervisor {}. (syn_score: {}, mem_score: {}, cpu_score: {})".format(
                        selected_dest,
                        host_to_score[selected_dest]["syn_score"],
                        host_to_score[selected_dest]["mem_score"],
                        host_to_score[selected_dest]["cpu_score"],
                    )
                )

                yield HostMigration(
                    name=vm_details["name"],
                    uuid=selected_vm["uuid"],
                    destination=selected_dest,
                    source_hypervisor=vm_details["OS-EXT-SRV-ATTR:hypervisor_hostname"],
                    status=MigrationStatus.to_be_migrated,
                )
            else:
                logging.warn(
                    "Could not find a hypervisor to fit {}".format(selected_vm["name"])
                )

        logging.debug(
            "Load balancer sleeping for {} seconds...".format(DELAY_BETWEEN_MIGRATIONS)
        )
        time.sleep(DELAY_BETWEEN_MIGRATIONS)


def do_load_balancing(conf, nova, dryrun):
    # type: (LoadbalancerConfig, OpenstackClient, bool) -> None
    """ This is the main loop.
        The basic strategy is:
            1. Get metrics
            2. Calculate an ordered list of a synthetic "load"
            3. Get list of VMs from the most loaded server
            4. Initiate a move of one of the VMs

    raises:
        NoHypervisorsFound
        AvailabilityZoneNotFound

        VMNotFound   (after move)
        VMNotActiveStatus  (after move)
        FlavourNotFound
        PrometheusError

        keystoneauth1.exceptions.http.Unauthorized
        keystoneauth1.exceptions.connection.ConnectFailure
        keystoneauth1.exceptions.connection.SSLError
        novaclient.exceptions.Forbidden
        novaclient.exceptions.BadRequest     (e.g. incompatible CPU spec)
    """
    logging.info("Starting load balancing (dry run mode: {})".format(dryrun))
    migrate_fn = dummy_migration if dryrun else vm_live_migration

    for vm_migration in create_vm_migrations(conf, nova):

        vm_migration = try_migrate_vm(nova, migrate_fn, vm_migration)

        # Check that the migration status is sane, and bomb out otherwise
        if vm_migration.status == MigrationStatus.vm_lost:
            # This is bad - the VM disappeared!
            # Raise an exception here, so that we don't accidentally kill the whole cluster...
            raise VMNotFound("VM {} cannot be found after move!".format(vm_migration))

        if vm_migration.status == MigrationStatus.vm_not_active:
            # Oops - this is probably pretty bad - did we kill it by moving it?
            # Raise an exception here, so that we don't accidentally kill the whole cluster...
            raise VMNotActiveStatus(
                "Tried to move VM with uuid {}, but it came back "
                "with a non active status in the end! Bailing out!".format(
                    vm_migration.uuid
                )
            )

        if vm_migration.status == MigrationStatus.vm_didnt_move:
            logging.warn(
                "Attempted to move VM with uuid {}, but after "
                "moving, it still exists on the same hypervisor".format(
                    vm_migration.uuid
                )
            )

        if vm_migration.status == MigrationStatus.invalid:
            # Something went wrong here - we can't find the Hypervisor or VM, for example
            logging.error(
                "Invalid state returned from migrate_fn for {}".format(
                    pprint.pformat(vm_migration)
                )
            )

        if vm_migration.status in [
            MigrationStatus.failed,
            MigrationStatus.vm_didnt_move,
        ]:
            logging.info(
                "Failed to move vm {} successfully, but not a fatal error.".format(
                    vm_migration.name
                )
            )


@click.command()
@click.option(
    "--dryrun",
    is_flag=True,
    help="Run in dryrun mode - output what hypervisors would have a host moved",
    default=False,
)
@click.option(
    "--configfile",
    help="Config file in 'dotenv' format.",
)
@click.option(
    "--debug",
    help="Dump debug messages to screen as well as log file",
    is_flag=True,
    default=False,
)
def main(dryrun, configfile, debug):
    """
    Openstack load leveller script

    This script attempts to "load balance" openstack, by live-migrating instances
    from busy hypervisors to less busy hypervisors.

    The busyness is calculated from the memory and cpu utilization (as reported by
    prometheus).

    The openstack credentials etc are stored in /etc/loadleveller-secrets.conf (by
    default).


    Exit codes:

        EX_SOFTWARE (70) if Prometheus can't be contacted, VMs are lost etc.

        EX_IOERR (74) for IO errors, specifically when reading in the config.

        EX_NOPERM (77) If openstack denies an action based on permissions.

        EX_CONFIG (78) For config errors, including requesting openstack resources that do not exist.

    """
    try:
        setup_logging(debug)
    except IOError as e:
        logging.error("Failed to set up logging: {}".format(e))
        sys.exit(os.EX_IOERR)  # EX_IOERR is 74

    try:
        conf = LoadbalancerConfig.load_config(configfile)
    except ConfigError as e:
        logging.error("{}".format(e))
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78

    nova = get_openstack_connection_object(conf)

    try:
        do_load_balancing(conf, nova, dryrun)
    except PrometheusError as e:
        logging.error("Issue getting data from Prometheus.")
        logging.error("Exception text: {}".format(e))
        sys.exit(os.EX_SOFTWARE)  # EX_SOFTWARE is 70
    except NoHypervisorsFound as e:
        logging.error(
            "No hypervisors found in availability zone. Perhaps check the limit_to_az config"
            "option (in the config file)"
        )
        logging.error("Exception text: {}".format(e))
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78
    except AvailabilityZoneNotFound as e:
        logging.error(
            "Availability zone specified in the limit_to_az config varible not found."
        )
        logging.error("Exception text: {}".format(e))
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78
    except VMNotFound as e:
        logging.error("VM not found after move. Exception text: {}".format(e))
        sys.exit(os.EX_SOFTWARE)  # EX_SOFTWARE is 70
    except VMNotActiveStatus as e:
        logging.error(
            "After a move, the VM is not active. Exception text: {}".format(e)
        )
        sys.exit(os.EX_SOFTWARE)  # EX_SOFTWARE is 70
    except FlavourNotFound as e:
        logging.error("Flavour of VM not found. Exception text: {}".format(e))
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78
    except (
        keystoneauth1.exceptions.http.Unauthorized,
        keystoneauth1.exceptions.connection.ConnectFailure,
    ) as e:
        logging.error(
            "Keystoneauth error against openstack. Check config. Exception: {}".format(
                e
            )
        )
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78
    except keystoneauth1.exceptions.connection.SSLError as e:
        logging.error(
            "SSL isue against the Openstack API. Consider the insecure parameter in "
            "the config file, or create signed certificates. Exception text: {}".format(
                e
            )
        )
        sys.exit(os.EX_CONFIG)  # EX_CONFIG is 78
    except novaclient.exceptions.Forbidden as e:
        logging.error(
            "The API user does not have the rights to perform a required action. Exception text: {}".format(
                e
            )
        )
        sys.exit(os.EX_NOPERM)  # EX_CONFIG is 77
    except novaclient.exceptions.BadRequest as e:
        logging.error(
            "Bad Request exception - this usually happens when the CPU spec of the hypervisors "
            "is incompatible. You can try turning on limit_to_exact_cpu_spec."
        )
        sys.exit(os.EX_SOFTWARE)  # EX_SOFTWARE is 70


if __name__ == "__main__":
    main()
