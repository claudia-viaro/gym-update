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
This is represented by a cross-sectional dataset with variables X_a, X_s, Y and N observations (# patients): {Y, X_a, X_s}_{i=1}^N  <br />
X_a, X_s follows a truncated normal distribution (a=0, b=inf) <br />
Y follows a Binomial distribution Bin(p, n), p=0.5 <br />

We are interested in observing the behaviour of ![equation](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5BY=1%7CX%5D=f(b_0%20&plus;%20b_1%20X_a%20&plus;%20b_2%20X_s))


#### The agent takes a step in the environment and:

Sees all patients and take an action a={theta_0, theta_1, theta_2} <br />
Runs a logistic regression on the patients using the action taken:  <br />
He computes the logit risk of each observation (\rho_1) <br />
He computes the g value, using \rho_1 and X_a <br />
He replaces the initial X_a with the g value, for each observation <br />
 <br />
He runs a logistic regression on the patients as a covariate has changed in value, retrieve new theta parameters (thetas2) <br />
He computes the logit risk of each observation (\rho_2) <br />
He computes the mean logit risk, which is the reward given by the environment back to the agent (as a result of the 'good deed' of the action) <br />
 <br />
The reward represents the mean hospitalization rate of the 'intervened' population of patients <br />
Then the episode ends and the environment resets <br />

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
