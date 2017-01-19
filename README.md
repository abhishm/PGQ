# PGQ

This paper proposes an approach to incorporate off-line samples in Policy Gradient. The authors were able to do this by drawing a parallel between Policy Gradient and Q-Learning. This is the second paper in ICLR 2017 that tries to use off-line samples to accelerate the learning in Policy Gradient. The summary of the first paper can be found [here](http://www.shortscience.org/paper?bibtexKey=journals/corr/WangBHMMKF16]). 

This is the first paper that describes the relationship between policy gradient and Q-learning. Lets consider a simple case of a grid world problem. In this problem, at every state, you can take one of the four actions: left, right, up, or down. Lets assume that, you are following a policy $\pi_\theta(\cdot|s_t)$. Let $\beta^{\pi_\theta}(s_t)$ denotes the stationary probability distribution of states under policy $\pi_\theta$ and $r(s, a)$ be the reward that we get after taking action $a$ at state $s$. Then our goal is to maximize
$$
\begin{array}{ccc}
&\arg\max_{\theta} \mathbb{E}_{s \sim \beta^{\pi_\theta}} \sum_{a}\pi_\theta(a|s) r(s, a) + \alpha H(\pi_\theta(\cdot| s))&\\
& \sum_{a} \pi_\theta(a|s) = 1 \;\;\forall\;\;  s
\end{array}
$$ 

Note that the above equation is a regularized policy optimization approach where we added a entropy bonus term, $H(\pi_{\theta}(\cdot | s))$ to enhance exploration. Mathematically,
$$
H(\pi_{\theta}(\cdot | s)) = - \sum_{a} \pi_\theta(a|s)\log\pi_\theta(a|s)
$$
and 
$$
\begin{array}{ccl}
\nabla_\theta H(\pi_{\theta}(\cdot | s))& = &-  \sum_{a} \left(1 + \log\pi_\theta(a|s)\right) \nabla_\theta \pi_\theta(a|s) \\
& = &- \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\left(1 + \log\pi_\theta(a|s)\right) \nabla_\theta \log\pi_\theta(a|s) \;\; \text{(likelihood trick)}
\end{array}
$$
Lagrange multiplier tells us that at the critical point $\theta^*$ of the above optimization problem, the gradient of optimization function should be parallel to gradient of constraint. Using the Policy Gradient Theorem, the gradient of the objective at optimal policy $\theta^*$ is 
$$
\mathbb{E}_{s \sim \beta^{\pi_{\theta^*}}, a \sim \pi_{\theta^*}(\cdot|s)} \nabla_{\theta} \log\pi_{\theta^*}(a|s) \left(Q^{\pi_{\theta^*}}(s, a) - \alpha  - \alpha \log\pi_{\theta^*}(a|s)\right)
$$ 

The gradient of the constraint at the optimal point $\theta^*$ is 
$$
\mathbb{E}_{s \sim \beta^{\pi_{\theta^*}}, a \sim \pi_{\theta^*}(\cdot|s)} \lambda_s  \nabla_{\theta^*}\log\pi_{\theta^*}(a|s)
$$

Using the theory of Lagrange multiplication
$$
\begin{array}{lll}
&&\mathbb{E}_{s \sim \beta^{\pi_{\theta^*}}, a \sim \pi_{\theta^*}(\cdot|s)} \nabla_{\theta}\log\pi_{\theta^*}(a|s) \left(Q^{\pi_{\theta^*}}(s, a) - \alpha  - \alpha \log\pi_{\theta^*}(a|s)\right) = \\
&&\mathbb{E}_{s \sim \beta^{\pi_{\theta^*}}, a \sim \pi_{\theta^*}(\cdot|s)} \lambda_s  \nabla_{\theta}\log\pi_{\theta^*}(a|s)
\end{array}
$$


If $\beta^{\pi_{\theta^*}}(s) > 0\;\; \forall\;\; s $ and $0 < \pi_{\theta^*}(a | s) < 1\;\; \forall\;\; s, a$, then for the tabular case of grid world, we get 
$$
Q^{\pi_{\theta^*}}(s, a) -  \alpha \log\pi_{\theta^*}(a|s) = \lambda_{s}\;\; \forall \;\; a 
$$
By multiplying both sides in above equation with $\pi_{\theta^*}(a|s)$ and summing over $a$, we get 
$$
\lambda_s = V^{\pi_{\theta^*}}(s) + \alpha H(\pi_\theta(\cdot | s))
$$
Using the value of Lagrange Multiplier, the action-value function of the optimal policy $\pi_{{\theta^*}}$ is
$$
{Q}^{\pi_{\theta^*}}(s, a) = V^{\pi_{\theta^*}}(s) + \alpha \left(H(\pi_{\theta^*}(\cdot | s)) + \log\pi_{\theta^*}(a|s)\right)
$$
and the optimal policy $\pi_{\theta^*}$ is a softmax policy 
$$
\pi_{\theta^*}(a|s) = \exp\left( \frac{Q^{\pi_{\theta^*}}(s, a) - V^{\pi_{\theta^*}}(s)}{\alpha} - H(\pi_{\theta^*}(\cdot|s))\right)
$$

The above relationship suggests that the optimal policy for the tabular case is a softmax policy of action-value function. Mainly;
$$
\pi_{\theta^*}(a|s) = \frac{\exp\left(\frac{Q^{\pi_{\theta^*}}(s,a)}{\alpha}\right)}{\sum_b \exp\left(\frac{Q^{\pi_{\theta^*}}(s,b)}{\alpha}\right)}
$$
Note that as the $\alpha \rightarrow 0$, the above policy become a greedy policy.
  
The authors suggest that even when the policy $\pi_{\theta}$ is not an optimal policy, we can still use the 
$\tilde{Q}^{\pi_\theta}(s, a)$ as an estimate for action-value of policy $\pi_\theta$ where  
$$
\tilde{Q}^{\pi_{\theta}}(s, a) = V^{\pi_{\theta}}(s) + \alpha \left(H(\pi_{\theta}(\cdot | s)) + \log\pi_{\theta}(a|s)\right)
$$
In the light of above definition of $\tilde{Q}^{\pi_\theta}(s, a)$, the update rule for the regularized policy gradient can be written as
$$
\mathbb{E}_{s \sim \beta^{\pi_\theta}, a \sim \pi_\theta(\cdot|s)} \nabla_{\theta} \log\pi_\theta(a|s) \left(Q^{\pi_\theta}(s, a) - {\tilde{Q}}^{\pi_\theta}(s, a) \right)
$$

**Author shows a strong similarity between Duelling DQN and Actor-critice method** 

In a duelling architecture, action-values are represented as summation of Advantage and Value function. Mathematically,
$$
Q(s, a) =  A^w(s, a) - \sum_a \mu(a|s) A(s, a) + V^\phi(s)
$$

The goal of the middle term in the above equation to obtain $A(s, a$ and $V(s)$ uniquely given $Q(s, a) \forall a$. In the Duelling architecture, we will update the $w$ and $\phi$ parameters as following:

$$
\begin{array}{ccc}
\Delta W &\sim& (r+ \max_b Q(s', b) - Q(s, a)) \nabla \left( A^w(s, a) - \sum_a \mu(a|s) A(s, a) \right) \\
\Delta \phi &\sim& (r+ \max_b Q(s', b) - Q(s, a)) \nabla \left( V(s) \right)
\end{array}
$$

Now assume an actor-critic approach, where policy is parametrized by $A^w(s, a)$ and there is a critic $V^\phi(s)$ of value-function. The policy $\pi_w(s, a)$ is 
$$
\pi_w(s, a) = \frac{e^{A^w(s, a)/\alpha}}{\sum_a e^{A^w(s, a)/\alpha}}
$$

Note that 
$$
\nabla_w \log\pi_w(s, a) = \nabla_w \frac{1}{\alpha}\left(A^w(s, a) - \sum_a \pi_w(s, a) A^w(s, a)\right)
$$

To estimate the Q-values of policy $\pi_w$, we are using the following estimator:
$$
\begin{array}{ccc}
Q(s, a) &=& \alpha \left(-\sum_a \pi_w(a | s)\log\pi_w(a | s) + \log\pi_w(a|s)\right) + V^\phi(s) \\
&=& A^w(s, a) - \sum_a \pi_w(a | s) A^w(s, a) + V^\phi(s)
\end{array}
$$

The actor update rule in the regularized policy gradient will be 
$$
\Delta W \sim (r+ \max_b Q(s', b) - Q(s, a)) \nabla \left( A^w(s, a) - \sum_a \pi_w(a|s) A^w(s, a) \right) 
$$
and the critic update rule is 
$$
\Delta \phi \sim (r+ \max_b Q(s', b) - Q(s, a)) \nabla V^\phi(s)
$$

Note that the rules to update $V^{\phi}$ is same in both DQN and actor-critic. The rule to update the critic varies in the probability distribution that is used to normalize the advantage function to make it unique.

** PGQ Algorithm**

Given this information PGQ Algorithm is simple and consist of the following steps:
1. Lets $\pi_\theta(a|s)$ represent our policy network and $V^w(s)$ represent our value network.
2. do $N$ times
   1. We collect samples $\{s_1, a_1, r_1, s_2, a_2, r_2, \cdots, s_T\}$ using policy $\pi_\theta$.
   2. We compute $Q^{\pi_\theta}(s_t, a_t) = \alpha(\log \pi(a_t| s_t) + H(\pi(\cdot|s_t))) + V(s_t)\;\; \forall t$. 
   3. We update $\theta$ using the regularized policy gradient approach:
   $$
   \begin{array}{ccc}
   \Delta \theta & = & \nabla_\theta \log \pi_\theta(a|s)(r_t + \max_b \tilde{Q}^{\pi_\theta}(s_{t+1}, b) - \tilde{Q}^{\pi_\theta}(s_{t}, a_{t} )\\
   \Delta \theta & = &  \nabla_\theta (W_\theta(s, a) - \sum_{a} \pi_\theta(a|s) W_\theta(s, a))(r_t + \max_b \tilde{Q}^{\pi_\theta}(s_{t+1}, b) - \tilde{Q}^{\pi_\theta}(s_{t}, a_{t} )
   \end{array}
   $$
   where $\pi_{\theta}(s, a) = \frac{e^{W(s, a)}}{\sum_b e^{W(s, b)}}$
  We update the critic by minimizing the mean square error:
   $$
   ((r_t + \max_b \tilde{Q}^{\pi_\theta}(s_{t+1}, b) - \tilde{Q}^{\pi_\theta}(s_{t}, a_{t} ))^2      
   $$

