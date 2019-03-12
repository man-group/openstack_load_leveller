# Openstack Load Leveller

The Openstack Load Leveller is a utility that will move VMs around hypervisors
until the load (as defined in the configuration) in terms of memory and CPU
usage reaches equilibrium.

The utility depends on having a Prometheus setup that can provide load metrics.


## Build / Install

1. `git clone https://github.com/manahl/openstack_load_leveller`
1. `cd openstack_load_leveller; pip install .`


## Configuration

Modify the configuration file (and optionally rename it) to reflect your
environment. It is recommended that you make the config file readable only by
the user you will be running the load leveller as. Also, if you prefer, you
can set the config options as environment variables instead.


### Configuring Openstack parameters

The leveller needs to be able to connect to the Openstack API. For this to work,
the following variables need to be defined:

```
OS_AUTH_URL:
    The Openstack API URL

OS_USERNAME, OS_DOMAIN_NAME, OS_PASSWORD:
    Credentials for an Admin user in Openstack
```


### Limiting moves to a specific AZ

Openstack itself places no limitations on live migrations into or out of
availability zones. This means that if you have more than one availability zone,
you probably want to set the below parameter, otherwise you will probably find
that VMs from one availability zone have migrated to other AZs after some time.

```
OS_LIMIT_TO_AZ:
    Only consider hypervisors in a particular availability zone when evaluating
    what to move.
```


### Configuring Prometheus parameters

The leveller needs a way of figuring out how loaded hosts are. The way this is
currently done is by querying Prometheus to get load stats (I.e. you will need a
Prometheus instance running in your environment).

The Prometheus options are described below:

```
PROMETHEUS_QUERY_URL:
    Base URL for prometheus queries

PROMETHEUS_MEM_USED:
    This query should return a normalised (0 to 1) number describing how much
    memory is used on the box. 0 = no memory used, 1 = all memory used.

PROMETHEUS_CPU_USED:
    Normalised CPU usage query (0 = no CPU used, 1 = all CPU used)

PROMETHEUS_METRIC_WEIGHT_MEM and PROMETHEUS_METRIC_WEIGHT_CPU:
    This is the relative weights assigned to the memory/CPU components when
    calculating the synthetic score.

    E.g if with PROMETHEUS_METRIC_WEIGHT_MEM=2, and
    PROMETHEUS_METRIC_WEIGHT_CPU=1, memory would be twice as significant in
    the synthetic metric calculation.

PROMETHEUS_MIN_LOAD_BEFORE_MOVE:
    Minimum load on the busiest hypervisor before a move is possible (i.e. if
    the Openstack cluster is lightly loaded, do nothing).

PROMETHEUS_MIN_LOAD_DIFF_BEFORE_MOVE:
    The minimum load difference betwen the highest and the lowest loaded
    hypervisor before a move is initiated, measured as a ratio between 0 and 1.

    This metric is calculated as: most_loaded_score / least_loaded_score - 1.0
```


## Running the tool

If you installed the tool with `pip`, you can simply run `openstacklb` from
the command line:

```
$ openstacklb --help
Usage: openstacklb [OPTIONS]

  Openstack load leveller script

  This script attempts to "load balance" openstack, by live-migrating
  instances from busy hypervisors to less busy hypervisors.

  The busyness is calculated from the memory and cpu utilization (as
  reported by prometheus).

  The openstack credentials etc are stored in /etc/loadleveller-secrets.conf
  (by default).

  Exit codes:

      EX_SOFTWARE (70) if Prometheus can't be contacted, VMs are lost etc.

      EX_IOERR (74) for IO errors, specifically when reading in the config.

      EX_NOPERM (77) If openstack denies an action based on permissions.

      EX_CONFIG (78) For config errors, including requesting openstack
      resources that do not exist.

Options:
  --dryrun           Run in dryrun mode - output what hypervisors would have a
                     host moved
  --configfile TEXT  Config file in 'dotenv' format.
  --debug            Dump debug messages to screen as well as log file
  --help             Show this message and exit.
```

Example:

```
$ openstacklb --configfile /etc/openstack-leveller.conf --dryrun --debug
```
