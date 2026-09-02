# Phase 11: Rotary Position Embeddings

This phase replaces learned absolute position embeddings with an optional
rotary position embedding (RoPE) path. The learned path remains the default so
the previous model behavior is preserved.

## Position Information in Attention

The learned baseline adds a trainable position vector to each token embedding
before the Transformer blocks. RoPE instead applies position-dependent
rotations to queries and keys inside each attention head. Values are not
rotated because they carry the content that the attention weights combine.

```text
learned: token embedding + learned position embedding -> blocks
RoPE:    token embedding -> rotate q and k inside attention
```

The configuration keeps these paths mutually exclusive. A RoPE model does not
construct an unused learned position table.

## Small Rotation Intuition

RoPE divides an even head dimension into adjacent feature pairs:

$$
(x_0,x_1),\ (x_2,x_3),\ (x_4,x_5),\ \ldots
$$

For position $m$ and pair frequency $\theta_i$, each pair is multiplied by the
2-by-2 rotation block:

$$
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
\begin{bmatrix}
x_{2i} \\
x_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
x_{2i}\cos(m\theta_i)-x_{2i+1}\sin(m\theta_i) \\
x_{2i}\sin(m\theta_i)+x_{2i+1}\cos(m\theta_i)
\end{bmatrix}
$$

For a $90^\circ$ rotation,
$\begin{bmatrix}1&0\end{bmatrix}^{\mathsf T}$ becomes
$\begin{bmatrix}0&1\end{bmatrix}^{\mathsf T}$. Its direction changes while its
magnitude remains one. In code, all pair blocks are evaluated together by
splitting even and odd features, applying the elementwise formulas, and
interleaving the rotated results.

## Why the Matrix Rotates the Vector

Write a two-dimensional vector in polar form:

$$
x=r\cos\phi, \qquad y=r\sin\phi
$$

Multiplying it by a rotation through angle $\alpha$ gives:

$$
\begin{aligned}
x' &= x\cos\alpha-y\sin\alpha, \\
y' &= x\sin\alpha+y\cos\alpha.
\end{aligned}
$$

Substituting the polar coordinates:

$$
\begin{aligned}
x'
&=r\cos\phi\cos\alpha-r\sin\phi\sin\alpha \\
&=r\cos(\phi+\alpha), \\
y'
&=r\cos\phi\sin\alpha+r\sin\phi\cos\alpha \\
&=r\sin(\phi+\alpha).
\end{aligned}
$$

The radius $r$ is unchanged, while the direction changes from $\phi$ to
$\phi+\alpha$. For RoPE pair $i$ at token position $m$, the added angle is
$\alpha=m\theta_i$.

The same proof can be written as an optional complex-number shorthand. Treat
the pair as

$$
z=x+iy=re^{i\phi}.
$$

Multiplication by the unit rotation gives

$$
ze^{i\alpha}=re^{i(\phi+\alpha)}.
$$

The implementation uses the equivalent real-valued sine and cosine formulas
rather than complex tensors.

## Full Higher-Dimensional Layout

For six features, the conceptual matrix contains three independent rotation
blocks:

$$
R_{\Theta,m}^{6}
=
\begin{bmatrix}
c_1 & -s_1 & 0 & 0 & 0 & 0 \\
s_1 & c_1 & 0 & 0 & 0 & 0 \\
0 & 0 & c_2 & -s_2 & 0 & 0 \\
0 & 0 & s_2 & c_2 & 0 & 0 \\
0 & 0 & 0 & 0 & c_3 & -s_3 \\
0 & 0 & 0 & 0 & s_3 & c_3
\end{bmatrix},
\qquad
c_i=\cos(m\theta_i),\quad s_i=\sin(m\theta_i).
$$

Multiplication rotates $(x_0,x_1)$, $(x_2,x_3)$, and $(x_4,x_5)$
independently. The zeros show why no pair mixes with another pair.

The code avoids constructing this mostly-zero matrix. Splitting even and odd
features aligns all first coordinates, all second coordinates, and their
corresponding cache columns:

```text
x_even = [x0,x2,x4]
x_odd  = [x1,x3,x5]
cosine = [c1,c2,c3]
sine   = [s1,s2,s3]
```

Elementwise multiplication then evaluates every nonzero 2-by-2 block at once.

## Frequencies and Tensor Shapes

For an attention-head width `D`, RoPE uses `D/2` frequencies:

$$
\theta_i=10000^{-2i/D},
\qquad i=0,1,\ldots,\frac{D}{2}-1.
$$

The cache combines every token position with every pair frequency:

```text
positions:   (T,)
frequencies: (D/2,)
angles:      (T,D/2)
cosine/sine: (T,D/2)
```

The cache broadcasts across the batch dimension because every example uses the
same rotation for a given token position.

## Relative Position Property

Let $R_m$ rotate a query at position $m$ and $R_n$ rotate a key at position
$n$. Their interaction is:

$$
\begin{aligned}
(R_mq)^{\mathsf T}(R_nk)
&=q^{\mathsf T}R_m^{\mathsf T}R_nk \\
&=q^{\mathsf T}R_{-m}R_nk \\
&=q^{\mathsf T}R_{n-m}k.
\end{aligned}
$$

The equality follows from $R_m^{\mathsf T}=R_{-m}$. Although each vector
receives an absolute-position rotation, their dot product depends on the signed
relative offset $n-m$. For example, $n-m=4$ means the key is four positions
after the query.

## Model Integration

The position choice flows through the component constructors:

```text
ModernModelConfig
    -> ModernTransformerBlock
    -> attention factory
    -> multi-head attention
    -> every single attention head
```

Only each single head stores the `use_rope` flag and acts on it during the
forward pass. The intermediate components only pass the construction choice to
their children.

## What the Tests Prove

The focused and acceptance tests verify:

- the cache matches manually calculated frequencies and angles;
- position zero is the identity rotation;
- pair shapes and magnitudes are preserved;
- the same cache broadcasts across batches;
- queries and keys are rotated separately while values remain unchanged;
- interactions with equal `n-m` offsets agree at different absolute positions;
- learned and RoPE configurations propagate to every attention head;
- a complete RoPE model produces finite logits, loss, and gradients;
- removing the learned table reduces parameters by
  $\text{block\_size}\times\text{d\_model}$.

These tests establish mathematical and integration correctness. They do not
show that RoPE improves validation loss or long-context generalization.

## What I Learned

- a high-dimensional RoPE operation is a collection of independent 2D
  rotations;
- the block-diagonal matrix can be implemented without materializing its zero
  entries;
- elementwise operations over the pair dimension perform every rotation at
  once;
- broadcasting preserves the batch dimension and reuses positional rotations
  across examples;
- RoPE changes query-key interactions before attention scores are calculated;
- constructor arguments can route one architectural choice to the small module
  that actually performs the operation;
- component tests, relative-position tests, and end-to-end tests prove
  different parts of the implementation.

## Open Questions

- How does RoPE compare with learned positions over several training seeds?
- How does the frequency base affect short and long positional patterns?
- When should caches be stored and reused instead of rebuilt each forward pass?
- How should position offsets be handled during cached autoregressive decoding?
