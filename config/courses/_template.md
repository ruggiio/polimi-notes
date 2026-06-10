# Scheda corso — <Nome Corso>

Copia questo file e rinominalo con lo slug del corso: nome in minuscolo,
ogni carattere non alfanumerico sostituito da `_`.
Esempio: "Model Order Reduction" → `model_order_reduction.md`

Tutto il contenuto della scheda viene iniettato nel prompt di generazione
delle note come guida di stile. La sezione `## Glossario` viene inoltre
usata come `initial_prompt` di faster-whisper per aiutare la trascrizione
dei termini tecnici (massimo ~800 caratteri utili).

## Glossario

- Proper Orthogonal Decomposition, POD
- proiezione di Galerkin
- snapshot matrix
- reduced order model, ROM

## Notazione

- Vettori in grassetto: $\mathbf{u}$, matrici maiuscole: $A$
- Lo spazio ridotto si indica con $V_N$, la base con $\{\zeta_i\}$

## Convenzioni LaTeX

- Usa gli ambienti `theorem`, `definition`, `corollary` del template
- Le dimostrazioni in ambiente `proof`, mai inline

## Stile del docente

- Il professore alterna italiano e inglese: uniforma le note in italiano,
  mantenendo i termini tecnici in inglese
