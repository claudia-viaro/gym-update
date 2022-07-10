# Gym-style API environment

## Environment dynamics

<img width="500" height="300" src="https://github.com/claudia-viaro/gym-update/blob/main/dynamics.png">

The functions used:
- $f_e(x^s, x^a) = \mathbb{E}[Y_e|X_e(1) = (x^s, x^a)]$: Causal mechanism determining probability of $Y_e = 1$ given $X_e(1)$. We will take $f_e(x^s, x^a) = (1 + \exp^{−x^s−x^a})^{−1}$
- $g^a_e(\rho, x^a) \in \{g : [0, 1] \times \Omega \rightarrow \Omega \}$: Intervention process on $X^a$ in response to a predictive score $\rho$ updating $X^a_e(0) \rightarrow X^a_e(1)$
- $\rho_e(x^s, x^a) \in \{\rho_e : \Omega^s \times \Omega^a \rightarrow [0, 1]\}$: Predictive score trained at epoch $e$


Additional information:
- At epoch $e$, the predictive score $\rho$ uses $X^a_e(0), X^s_e(0)$ and $Y_e$ as training data; previous epochs are ignored and $X^a_e(1), X^s_e(1)$ are not observed. The predictive score is computed at time $t=0$.
- We allow $\rho_e$ to be an arbitrary function, but generally presume it is an estimator of $\rho_e(x^s, x^a) \approx E [Y_e|X^s_e(0) = x^s, X^a_e(0) = x^a]= f_e(x^s, g^a_e(\rho_{e-1}, x^a)) \triangleq \tilde{f}_e(x^s, x^a) $
- $\forall e f_e = E[Y_e|X_e] = E[Y_e|X_e(1)]$: $Y_e$ depends on $X_e(1)$; that is, after any potential interventions
- a higher value $\rho$ means a larger intervention is made (we assume $g^a_e$ to be deterministic, but random valued functions may more accurately capture the
uncertainty linked to real-world interventions)



## Naive updating
By  ‘naive’ updating it is meant that a new score $ρ_e$ is fitted in each epoch, and then used as a drop-in replacement of an existing score $ρ_{e−1}$. It leads
to estimates $\rho_e(x^s, x^a)$ converging as $e \rightarrow \infty$ to a setting in which $\rho_e$ accurately estimates its own effect: conceptually, $\rho_e(x^s, x^a)$ estimates the probability of $Y$ after interventions have been made on the basis of $\rho_e(x^s, x^a)$ itself. <br /> 

**EPOCH 0** <br />
**t=0** <br />
- observe a population of patients $(X_0^a(0),X_0^s(0))_{i=1}^N$

**t=1** <br />
- there are no interventions, hence $X_0^a(1) = X_0^a(0)$
- the risk of observing $Y = 1$ depends only on covariates at $t1$ through $f_0$ and is $E[Y_0|X_0(0) = (x^s, x^a)] =f(x^s, x^a)$
- the score $\rho_0$ is therefore defined as $\rho_0(x^s, x^a) = f(x^s, x^a)$
- $Y_0$ is observed 
- analyst decides a function $\rho_0$, which is retained into epoch 1. We will use initialized actions $&theta = (&theta^0, \theta^1, \theta^2)$

_The model performance under non-intervention is equivalent to performance at epoch 0_ <br />

**EPOCH $>0$**
**t=0**<br />
- observe a new population of patients $(X_e^a(0),X_e^s(0))_{i=1}^N$
- analyst computes $\rho_0 (X^s_e(0), Xa_e(0))$

**t=1**<br />
- $X^s_e(0)$ is not interventionable and becomes $X^s_e(1)$
- $\rho_0$ is used to inform interventions $g^a_e$ to change values $X^a_e(1) = g_e(\rho_{e-1}(x^s, x^a), x^a)$
- $E[Y_1]$ is determined by covariates $X^s_e(1), X^a_e(1)$
- the score $ρ_e$ is defined as $\rho_e(x^s, x^a) = f(x^s, g^a(\rho_{e−1}(x^s, x^a), xa)) \triangleq h(\rho_{e−1}(x^s, x^a))
- $Y_e$ is observed 
- analyst decides a function $\rho_e$ using $X^s_e(1), X^a_e(1), Y_e$, which is retained into epoch $e+1$. We will use $\rho_e =(1 + e^{−\theta^0 −x^s \theta^1 −x^a \beta^2 )^{−1}$ <br />

Then the episodes repeat <br />

## state and action spaces:
Action space: 3D space $\in [-2, 2]$. Actions represent the coefficients thetas of a logistic regression that will be run on the dataset of patients         <br />    

Observation space: 2D space $\in [0, \infty)$. States represent values for the covariates $X^a, X^s$  <br />

## To install
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
