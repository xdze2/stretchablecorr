
# Finite Strain Theory

ref. [Applied Mechanics of Solids, Allan F. Bower](http://solidmechanics.org/)


- **Displacement field $u$ :**

$$
x(t) = x_0 + u(x_0, \, t)
$$

where $x_0$ is the reference position and $x$ the new position. 

- **Displacement Gradient Tensor :**

$$
u \otimes \nabla_{ik} = \frac{\partial u_i}{\partial x_k}
$$

Indices, $i$ and $k$, representes spatial dimensions.

--> Finite difference, could be point-centered (3x3 points) or cell-centered (2x2 points)


- **Deformation Gradient Tensor :**

$$
F = I + u \otimes \nabla
$$


Volume change is given by The Jacobian :

$$
J = det(F)= \frac{dV}{dV_0}
$$


- **The Lagrange strain tensor :**

$$
E = \frac{1}{2}(\, FF^T - I \,)
$$