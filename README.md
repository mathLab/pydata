<p align="center">
    <a href="https://github.com/mathLab/EZyRB/blob/master/LICENSE.rst" target="_blank">
        <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
    </a>
    <a href="https://travis-ci.org/mathLab/pydata" target="_blank">
        <img alt="Build Status" src="https://travis-ci.org/mathLab/pydata.svg">
    </a>
    <a href="https://coveralls.io/github/mathLab/pydata" target="_blank">
        <img alt="Coverage Status" src="https://coveralls.io/repos/github/mathLab/pydata/badge.svg">
    </a>
    <a class="badge-align" href="https://www.codacy.com/manual/mathLab/pydata?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mathLab/pydata&amp;utm_campaign=Badge_Grade">
        <img src="https://api.codacy.com/project/badge/Grade/ff6570910f5e4101ad4fa6c0700a07db"/>
    </a>
</p>


**pydata** is a python package that allows you to handle several scientific file formats.

## Table of contents
* [Dependencies and installation](#dependencies-and-installation)
* [Testing](#testing)
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
* [License](#license)

## Dependencies and installation
**pydata** requires only `numpy`, `scipy`, `matplotlib`. Other packages may be required in order to manipulate specific files. We provide a simple table to summarize the needed packages for supported files:

| Extension | Package required |
|-----------|------------------|
| `.vtk`    | `vtk`            |
| `.stl`    | `vtk`            |
| `.iges`   | `python-occ`     |



The official distribution is on GitHub, and you can clone the repository using

```bash
> git clone https://github.com/mathLab/pydata
```

To install the package just type:

```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```

## Testing
We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/pydata).

To run tests locally (the package `nose` is required):

```bash
> python test.py
```

## Authors and contributors
**pydata** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Marco Tezzele](mailto:marcotez@gmail.com)
* [Nicola Demo](mailto:demo.nicola@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it).

Contact us by email for further information or questions about **pydata**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## How to contribute
We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
