# .XIN @ BERLIN CLASSICS 2019 WORKSHOP #

[Workshop Program](https://www.mpiwg-berlin.mpg.de/event/digital-humanities-and-classical-studies-prospects-and-challenges)

Three papers on the **CTEXT** data set:
	- Slingerland
	- Nichols
	- Nielbo

This repository contains code and paper for Nielbo. Code is only for demonstration purpose.


## Replication ##

```bash

cd src
./main.sh

```

Notice that the training lexical model will utilize all available cores to run a grid search for hyper-parameters, which is resource/time consuming. Consider using the provided model and only running the individual scripts.

## Presentation ##

```bash

cd paper
pdflatex main.tex

```

## Data ##

`shangshu` contains the* Shangshu* in four sub-directories for age-based prediction  
`slingerland_coprus` contains the full *CTEXT* corpus for

	- Slingerland, E., Nichols, R., Nielbo, K., & Logan, C. (2017). The Distant Reading of Religious Texts: A Big Data Approach to Mind-Body Concepts in Early China. Journal of the American Academy of Religion, 85(4), 985–1016.
	- Nichols, R., Slingerland, E., Nielbo, K., Bergeton, U., Logan, C., & Kleinman, S. (2018). Modeling the Contested Relationship between Analects, Mencius, and Xunzi: Preliminary Evidence from a Machine-Learning Approach. The Journal of Asian Studies, 77(01), 19–57.
