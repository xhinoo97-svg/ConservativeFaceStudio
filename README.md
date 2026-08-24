# Conservative Face Studio

Applicazione Windows offline per il restauro fotografico conservativo, progettata per funzionare in modalità CPU sicura e usare accelerazione soltanto dopo un test reale del driver.

## Pipeline production

La foto numero 1 resta sempre la MAIN IMAGE e il canvas finale. Il programma accetta da 0 a 9 reference e applica una pipeline verificabile:

1. import immutabile, preflight e analisi;
2. allineamento globale o locale per reference parziali;
3. damage/recoverability, Evidence Atlas e component bank;
4. riparazione reference-first, restoration e fusion con rollback;
5. validazione di provenance/confidence ed export di `final_restored_main.png`.

Il pacchetto production include YuNet, SFace, NAFNet ONNX, face parsing, head pose e LaMa. I modelli di ricerca o pesanti restano disabilitati e non possono essere selezionati come backend production. CPU, RAM, GPU e driver non vengono mai presunti: il profilo hardware è misurato a runtime e il fallback CPU resta sempre disponibile.

## Installazione Windows

Scarica `ConservativeFaceStudio-Setup-x64.exe`, installa e avvia dal collegamento. L'installer contiene runtime e pesi production: non servono Python, pip, Git, terminale o download al primo utilizzo.

Il pacchetto `ConservativeFaceStudio-Windows-x64.zip` offre la stessa applicazione in forma portable.

## Uso

1. Carica la MAIN IMAGE.
2. Aggiungi facoltativamente fino a 9 reference.
3. Avvia la pipeline.
4. Salva lo ZIP dei risultati oppure il progetto recuperabile.

La restoration normale non usa la rete. Il controllo aggiornamenti è esplicito e separa app e modelli; ogni download viene verificato con SHA-256, smoke test e rollback atomico.

## Avvio da sorgente per sviluppo

```bat
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m app
```

Oppure avvia `run_windows.bat`.

## Verifica pacchetto

L'eseguibile supporta `--smoke-test`, `--verify-installation` e `--offline-test`. La release Windows esegue questi controlli sia sulla cartella portable sia dopo un'installazione silenziosa in una directory pulita.
