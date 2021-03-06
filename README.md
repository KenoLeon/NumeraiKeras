# NumeraiKeras 🧠

**Keras-Tensorflow boilerplate starter for the [Numerai][d3d768b1]
 competition.**

  [d3d768b1]: https://numer.ai/ "Numerai"


## What is it ?

I wanted to use [Keras][12781e27] for the Numerai tournament and couldn't find a basic example  so I made this one.

  [12781e27]: https://keras.io/ "Keras.io"


## Medium Article :

  If you are new to the competition and/or want a better explanation of the code see this companion post:

  [Practical Keras: Simple regression for the Numerai Tournament ][a0f97d78]


  [a0f97d78]: https://medium.com/@k3no/practical-keras-59c9d18ef6cf "Medium Article"



## Validation/Metrics:

There's a separate script to calculate Validation Correlation ([val_corr.py][cf7ec790]) which should match the one generated by Numerai.

  [cf7ec790]: https://github.com/KenoLeon/NumeraiKeras/blob/master/val_corr.py "validation correlation"


And some barebone metrics ([metrics.py][3f432b7b]) contributed by [parmarsuraj99][790a9a80] ( [see also this forum post][86678643] )

  [3f432b7b]: https://github.com/KenoLeon/NumeraiKeras/blob/master/metrics.py "metrics.py"
  [790a9a80]: https://github.com/parmarsuraj99 "parmarsuraj99"
  [86678643]: https://forum.numer.ai/t/more-metrics-for-ya/636 "metrics numerai"


## Caveat Emptor (Buyer Beware)
Out of the box this NN/ Model is not very good, you need to tune parameters, add your preferred validation, tweak the model, etc, etc  see the following section...   


## Contribute - To Do - Upgrades:

Open an issue or pm on the Numerai chat if you want to add your contributions.

Ideally each new contribution is a standalone script with a new feature, some missing features :

- Save the model ( as to not retrain )
- Cross validation ( within the dataset and with val1 val2 )
- Payout scores and metrics (see Validation)
- Parameter tuning.
- Colab/jupyter notebook.
- Your suggestion ?

It would really be cool to have a community model to go against example_predictions, just saying...



## Feeling generous ?
Toss a 💰to your 🧙‍er !

**NMR/ETH:** 0xD2468b070e27138525ae347410A5cFFcEDc3d0E7
