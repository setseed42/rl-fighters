from setuptools import setup

setup(name='gym_fighters',
      version='0.0.1',
      include_package_data=True,
      #packages=['mypkg'],
      #package_dir={'mypkg': 'src/mypkg'},
      package_data={'gym_fighters': ['gym_fighters/envs/*.png']},
      install_requires=['gym==0.15.3', 'numpy', 'pyglet==1.3.2']#And any other dependencies required
)
