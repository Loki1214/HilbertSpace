# HilbertSpace

## Coding convensions
- Naming convensions <br>

| | First letter | Word concatenation |
| :--- | :---: | ---: |
| Variable   | Lowercase | Camel case |
| Type alias | Uppercase | Camel case |
| Class      | Uppercase | Camel case |
| Method     | Lowercase | Snake case |
| Function   | Lowercase | Snake case |

- Abbreviations

| Word | Abbreviations |
| :--- | :---: |
| system size   | sysSize |
| momentum | k |

## Coding conventions for mathematical objects

- Correspondence between Fock states and array objects <br>
![Correspondence between Fock states and array objects](.Fig_README/FockToArray.png)


- Correspondence between Fock states and integers <br>
![Correspondence between Fock states and array objects](.Fig_README/FockToInteger.png) <br>
where $d_{\mathrm{loc}} \coloneqq \dim \mathcal{H}_{\mathrm{loc}}$. <br>
- The type of $\mathrm{stateNum}$ should be **Size (= Eigen::Index)**.

- Translation operator: <br>
![Translation operator](.Fig_README/TranslationOp.png)
