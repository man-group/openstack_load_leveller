
from setuptools import setup

setup(name='openstacklb',
      version='0.1',
      description='The openstack load leveller',
      url='http://github.com/manahl/openstack_load_leveller',
      author='MAN Alpha Tech',
      author_email='ManAHLTech@ahl.com',
      license='GPL-2',
      packages=['openstacklb'],
      zip_safe=False,
      install_requires=['python-dotenv==0.9.1',
                        'python-novaclient==11.1.0',
                        'attrs==18.2.0',
                        'enum==0.4.7',
                        'urllib3==1.24',
                        'click==7.0',
                        'python-novaclient==11.1.0'],

      entry_points = {
         'console_scripts': ['openstacklb=openstacklb.command_line:main'],
      }

      )

