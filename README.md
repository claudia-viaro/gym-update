# Gym-style API environment

## Environment dynamics

<img width="500" height="300" src="https://github.com/claudia-viaro/gym-update/blob/main/dynamics.png">

The functions used:
- $f_e(x^s, x^a) = \mathbb{E}[Y_e|X_e(1) = (x^s, x^a)]$: Causal mechanism determining probability of $Y_e = 1$ given $X_e(1)$
- $g^a_e(\rho, x^a) \in \{g : [0, 1] \times \Omega \rightarrow \Omega \}$: Intervention process on $X_a$ in response to a predictive score $\rho$ updating $X^a_e(0) \rightarrow X^a_e(1)$
- $\rho_e(x^s, x^a) \in \{\rho_e : \Omega^s \times \Omega^a \rightarrow [0, 1]\}$: Predictive score trained at epoch $e$


Additional information:
- At epoch $e$, the predictive score $\rho$ uses $X^a_e(0), X^s_e(0)$ and $Y_e$ as training data; previous epochs are ignored and $X^a_e(1), X^s_e(1)$ are not observed. The predictive score is computed at time $t=0$.
- We allow $\rho_e$ to be an arbitrary function, but generally presume it is an estimator of $\rho_e(x^s, x^a) \approx \mathbb{E} [Y_e|X^s_e(0) = x^s, X^a_e(0) = x^a]= f_e(x^s, g^a_e(\rho_{e−1}, x^a))] \triangleq \tilde{f}_e(x^s, x^a)$
- $\forall e f_e = \mathbb{E}[Y_e|X_e] = \mathbb{E}[Y_e|X_e(1)]$: $Y_e$ depends on $X_e(1)$; that is, after any potential interventions
- a higher value $\rho$ means a larger intervention is made (we assume $g^a_e$ to be deterministic, but random valued functions may more accurately capture the
uncertainty linked to real-world interventions)



## state and action spaces:
Action space: 3D space $\in [-2, 2]$. Actions represent the coefficients thetas of a logistic regression that will be run on the dataset of patients         <br />    

Observation space: 2D space $\in [0, \infty)$. States represent values for the covariates $X_a, X_s$  <br />


At every episode there is a new population of patients, it is represented by a cross-sectional dataset ![equation](https://latex.codecogs.com/svg.image?%5C%7BY,%20X_a,%20X_s%5C%7D_%7Bi=1%7D%5EN)  <br />
![equation](https://latex.codecogs.com/svg.image?20X_a,%20X_s) follows a truncated normal distribution ![equation](https://latex.codecogs.com/svg.image?(a=0,%20b=%5Cinfty)) <br />
![equation](https://latex.codecogs.com/svg.image?Y) follows a Binomial distribution ![equation](https://latex.codecogs.com/svg.image?Bin(p,%20n)), ![equation](https://latex.codecogs.com/svg.image?p=0.5) <br />

We are interested in observing the behaviour of ![equation](https://latex.codecogs.com/svg.image?%5Cmathbb%7BE%7D%5BY=1%7CX%5D=f(b_0%20&plus;%20b_1%20X_a%20&plus;%20b_2%20X_s))


#### The environment produces the following iteration:
<br />
**e=0, t=0**<br />
Sees a population of patients ![equation](https://latex.codecogs.com/svg.image?(Y,&space;X_a(0),&space;X_s(0))_{i=1}^N) <br />
<br />
**e=0, t=1**<br />
See the same population ![equation](https://latex.codecogs.com/svg.image?(Y,&space;X_a(1),&space;X_s(1))_{i=1}^N) <br />
take an action ![equation](https://latex.codecogs.com/svg.image?a=%5C%7B%5Ctheta_0,%20%5Ctheta_1,%20%5Ctheta_2%5C%7D) <br />
Computes the logit risk of each observation ![equation](https://latex.codecogs.com/svg.image?\rho_1(X_a(1),&space;X_s(1))) <br />
<br />
**e=1, t=0**<br />
See a new population of patients ![equation](https://latex.codecogs.com/svg.image?(Y,&space;X_a(0),&space;X_s(0))_{i=1}^N) <br />
Computes the intervention value ![equation](https://latex.codecogs.com/svg.image?%5Cbar%7BX%7D_a%20=%20g(%5Crho_1,%20X_a)) <br />

**e=1, t=1**<br />
See outcome Y <br />
Fits a logistic regression on the patients: ![equation](https://latex.codecogs.com/svg.image?\mathbb{E}[Y=1|\bar{X}_a,&space;X_s]=\frac{1}{1&plus;&space;\exp^{-(\theta_0&plus;\theta_1&space;\bar{X_a}&plus;\theta_2&space;X_s)}) <br />
Retrieves the coefficients ![equation](https://latex.codecogs.com/svg.image?\{\theta_0^{(2)},&space;\theta_1^{(2)},&space;\theta_2^{(2)}\}) <br />
Computes the logit risk of each observation ![equation](https://latex.codecogs.com/svg.image?%5Crho_2)  <br />
Computes the mean logit risk across all observations, which is the reward given by the environment back to the agent (as a result of the 'good deed' of the action) <br />

Then the episode ends and the environment restarts from episode 1 <br />

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
