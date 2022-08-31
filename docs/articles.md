## General Mkdocs

1. Footnotes should follow punctuation: https://www.bristol.ac.uk/arts/exercises/referencing/page_08.htm#:~:text=Footnote%20or%20endnote%20numbers%20in,the%20end%20of%20a%20sentence.
2. ${\color{red} x} + {\color{blue} y}$ for color latex.


## Latex/Mathjax

$$\newcommand{\F}{\mathbb{F}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\v}{\mathbf{v}}
\newcommand{\a}{\mathbf{a}}
\newcommand{\b}{\mathbf{b}}
\newcommand{\c}{\mathbf{c}}
\newcommand{\w}{\mathbf{w}}
\newcommand{\u}{\mathbf{u}}
\newcommand{\0}{\mathbf{0}}
\newcommand{\1}{\mathbf{1}}$$

!!! info
    All the mathjax typesets are placed in `mathjax.js`.

### Label Equations

We label equations by using $\eqref{eq:sample}$, we find the value of an
interesting integral:

$$
\begin{equation}
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
  \label{eq:sample}
\end{equation}
$$

Note that it is necessary to use the `label` command to label equations, and it is also
important to give an indicative name for the label, so that the label can be used in references.
This label method supports automatic numbering, as you can see in the example.

```
tex: {
    tags: 'ams'
}
```

---

### Color

To add color code:

```
loader: { load: ['[tex]/color'] },
tex: { packages: { '[+]': ['color'] } }
```

${\color{red} x} + {\color{blue} y}$

!!! warning
    Note that since `tags: 'ams'` is a key-pair inside `tex`, then we should instead write:

    ```
    loader: { load: ['[tex]/color'] },
    tex: { packages: { '[+]': ['color'] }, tags: 'ams'}
    ``` 