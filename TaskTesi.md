# TASK TESI

- [ ] Analizzare differenze tra le edge map a diverse orientazioni del CORF e le cellule B create con l’edge map completa e orientations matrix.

- [ ] Se nel punto 1 non vi sono differenze,utilizzare gli eventi per poter generare edge map a differenti orientazioni utilizzando, inizialmente, la DOG multivariata per ottenere le prime B(feedforward). Se nel punto 1 vi sono differenze, creare l’orientations matrix assumento che le telecamere ad eventi ci forniscano una edge map e creare le prime B(feedforward).

- [ ] Confrontare i risultati ottenuti nel punto 2 con il modello originale(utilizzare immagine RGB da cui sono stati estratti gli eventi).

- [ ] Implementare le equazioni delle cellule B(feedback) e G tenendo conto che si stanno utilizzando gli eventi(usare image pos e neg dove sono presenti le cellule S nelle equazioni).

- [ ] Testare l’intero modello e analizzare i risultati.

- [ ] Testare alcune proprietà del CORF(contrast invariant orientation tuning, per esempio) nel modello implementato in modo da capire se riesce a “manifestarle”.

- [ ] Se i test al punto 6 falliscono, utilizzare il CORF con gli eventi.