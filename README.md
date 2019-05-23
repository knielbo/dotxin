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