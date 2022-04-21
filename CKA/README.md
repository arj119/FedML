# Centered Kernel Alignment CKA

CKA measures the similarity between feature layers of a neural network. Where 0 means no similarity and 1 is highest 
similarity.

As per Kornblith et al. (2019), CKA is shown in Eq 3. Here, the $i^{th}$ is $\lambda^i_X$

Linear CKA:

$$ CKA(XX^T, YY^T) = \frac{||Y^T X||^2_F}{||X^TX||_F||Y^TY||_F} $$