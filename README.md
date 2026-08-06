# Conservative Face Studio

Applicazione Windows a blocchi per il restauro fotografico conservativo, progettata per funzionare principalmente su CPU.

## Obiettivo iniziale

La prima versione implementa una pipeline verificabile:

1. Importazione della foto principale e dei riferimenti.
2. Deblur e denoise con anteprima.
3. Accetta, riprova o salta il blocco.
4. Miglioramento di contrasto e nitidezza.
5. Esportazione finale in PNG o JPEG.

I moduli AI più pesanti verranno integrati come plugin separati, per evitare conflitti tra dipendenze e mantenere l'app utilizzabile su HP EliteBook x360 1030 G3.

## Avvio su Windows

```bat
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m app
```

Oppure avvia `run_windows.bat`.

## Stato

Progetto inizializzato. Il primo traguardo è un'app funzionante con workflow a blocchi e download/esportazione della foto finale.
