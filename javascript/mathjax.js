window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    },

    loader: { load: ['[tex]/color'] },
    tex: { packages: { '[+]': ['color'] }, tags: 'ams' },

};

document$.subscribe(() => {
    MathJax.typesetPromise()
});



