# Scheda corso — MACHINE LEARNING FOR MECHANICAL SYSTEMS

Machine learning per sistemi meccanici, Politecnico di Milano, A.A. 2026/27.

## Glossario

- supervised learning, unsupervised learning, regression, classification, clustering, overfitting, regularization
- training set, validation set, test set, cross-validation, bias-variance tradeoff, loss function, gradient descent
- linear regression, logistic regression, support vector machine, SVM, decision tree, random forest, gradient boosting
- neural network, backpropagation, activation function, ReLU, convolutional neural network, CNN, recurrent, LSTM, autoencoder
- principal component analysis, PCA, feature extraction, feature engineering, time series, condition monitoring
- fault detection, fault diagnosis, predictive maintenance, remaining useful life, RUL, anomaly detection, digital twin
- Gaussian process, Bayesian, kernel, hyperparameter, learning rate, batch size, epoch, dropout

## Notazione

- Dati $\mathbf{x}_i \in \mathbb{R}^d$, etichette $y_i$, parametri $\boldsymbol{\theta}$, funzione di costo $J(\boldsymbol{\theta})$
- Matrici in maiuscolo $X$, vettori in grassetto $\mathbf{w}$

## Stile del docente

- Ogni algoritmo: idea intuitiva → formulazione → passi dell'algoritmo → esempio meccanico

## Derivazioni

Quando la lezione deriva qualcosa (funzione di costo, gradiente, regola di aggiornamento,
backpropagation, soluzione in forma chiusa…), il valore degli appunti sta nei passaggi.

- Riporta OGNI passaggio intermedio che il docente scrive, mostra sulle slide o dice a voce
  (regola della catena, derivata di un termine, sostituzione, cambio di indici): non fondere
  più uguaglianze in una sola e non saltare dalla definizione al risultato.
- Per ogni passaggio non ovvio scrivi in una riga di prosa che cosa si è fatto e perché.
- Ogni equazione con un ruolo nel ragionamento va in un ambiente display numerato; definisci
  le quantità che compaiono, con le dimensioni (vettori in ℝ^d, matrici n×d, pesi di uno
  strato…) quando il docente le dice.
- Le parti di lezione su codice, librerie o esempi pratici NON vanno trasformate in
  derivazioni: lì riporta il codice e il ragionamento del docente così come sono.
- Solo le derivazioni che il docente ha fatto davvero: non aggiungere analisi, formule o
  esempi numerici di tuo, nemmeno se corretti e utili. Completare un passaggio che il
  docente ha saltato va bene; estendere il ragionamento oltre quello che ha detto, no.
