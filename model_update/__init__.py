#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from gym.envs.registration import register

register(
    id='m.update-v0',
    entry_point='gym_m.update.envs:ModelUpdateEnv',
)

