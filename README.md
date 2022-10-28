# HilbertSpace

## Coding conventions for mathematical objects

- Correspondence between Fock states and array objects
$$
	\ket{\alpha_{1}\cdots \alpha_{L-1}\alpha_{L}}
	\longleftrightarrow a[j] = \alpha_{j+1} \quad(j=0,\dots,L-1).
$$


- Correspondence between Fock states and integers
$$
	\ket{\alpha_{1}\cdots \alpha_{L-1}\alpha_{L}}
	\longleftrightarrow \sum_{j=0}^{L-1} \alpha_{j+1} \ (d_{\mathrm{loc}})^{j},
$$
where $d_{\mathrm{loc}} \coloneqq \dim \mathcal{H}_{\mathrm{loc}}$.

- Translation operator:
$$
	\hat{T}_{L} \ket{\alpha_{1}\cdots \alpha_{L-1}\alpha_{L}} \coloneqq \ket{\alpha_{L}\alpha_{1}\cdots \alpha_{L-1}}
$$
