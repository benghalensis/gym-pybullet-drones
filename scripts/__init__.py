from gym.envs.registration import register
from RacingDroneAviary import RacingDroneAviary

register(
  id='RacingDroneAviary-v0',
  entry_point='RacingDroneAviary:RacingDroneAviary'
)