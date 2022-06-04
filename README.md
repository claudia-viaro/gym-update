# Gym-style API environment


#### The domain features a continuos state and action space:
Action space: ```
self.action_space = spaces.Box(
                                      low = np.float32(-np.array([2, 2, 2])),
                                      high = np.float32(np.array([2, 2, 2])))```
*the actions represent the coefficients thetas of a logistic regression that will be run on the dataset of patients            

Observation space: `self.observation_space = spaces.Box(
                                                low=np.array([0], 
                                                high=np.array([1], 
                                                dtype=np.float32)  `          
*the states represent values for the covariates X_a, X_s

#### The environment resets:

New population of patients at every episode <br />
This is represented by a cross-sectional dataset with variables ![equation](https://latex.codecogs.com/svg.image?Y,%20X_a,%20X_s) and ![image](https://latex.codecogs.com/svg.image?N) observations (# patients): ![image](https://latex.codecogs.com/svg.image?%5C%7BY,%20X_a,%20X_s%5C%7D_%7Bi=1%7D%5EN)  <br />
![equation](https://latex.codecogs.com/svg.image?20X_a,%20X_s) follows a truncated normal distribution ![image](https://latex.codecogs.com/svg.image?(a=0,%20b=%5Cinfty)) <br />
![equation](https://latex.codecogs.com/svg.image?Y) follows a Binomial distribution ![equation](https://latex.codecogs.com/svg.image?Bin(p,%20n)), ![equation](https://latex.codecogs.com/svg.image?p=0.5) <br />

We are interested in observing the behaviour of ![equation](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5BY=1%7CX%5D=f(b_0%20&plus;%20b_1%20X_a%20&plus;%20b_2%20X_s))


#### The agent takes a step in the environment and:

Sees all patients and take an action !image](https://latex.codecogs.com/svg.image?a=%5C%7B%5Ctheta_0,%20%5Ctheta_1,%20%5Ctheta_2%5C%7D) <br />
Fits a logistic regression on the patients using the action taken: ![image](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5BY=1%7CX%5D=%5Cfrac%7B1%7D%7B1&plus;%20%5Cexp%5E%7B-(X_s&plus;X_a)%7D) <br />
Computes the logit risk of each observation ![equation](https://latex.codecogs.com/svg.image?%5Crho_1) <br />
Computes the intervention value ![equation](https://latex.codecogs.com/svg.image?%5Cbar%7BX%7D_a%20=%20g(%5Crho_1,%20X_a)) <br />
Fits a new model [Replaces the initial X_a with the g value, for each observation](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5BY=1%7C%5Cbar%7BX%7D_a,%20X_s%5D=logistic(%5Cbar%7BX%7D_a,%20X_s)) <br />
 <br />
Retrieves the coefficients ![equation](https://latex.codecogs.com/svg.image?\{\theta_0^{(2)},&space;\theta_1^{(2)},&space;\theta_2^{(2)}\}) <br />
Computes the logit risk of each observation ![equation](https://latex.codecogs.com/svg.image?%5Crho_2)  <br />
Computes the mean logit risk across all observations, which is the reward given by the environment back to the agent (as a result of the 'good deed' of the action) <br />
 <br />
The reward represents the mean hospitalization rate of the 'intervened' population of patients <br />
Then the episode ends and the environment resets to a new population. <br />

# To install
- git clone https://github.com/claudia-viaro/gym-update.git
- cd gym-update
- !pip install gym-update
- import gym
- import gym_update
- env =gym.make('update-v0')

# To change version
- change version to, e.g., 1.0.7 from setup.py file
- git clone https://github.com/claudia-viaro/gym-update.git
- cd gym-update
- python setup.py sdist bdist_wheel
- twine check dist/*
- twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
