# Moduli esterni e politica dei modelli

| Modulo | Funzione | Licenza codice | Politica pesi | Stato |
|---|---|---|---|---|
| OpenCV | deblur, denoise, colore, maschere, blending | Apache-2.0 | nessun peso | integrato nel core |
| MediaPipe Face Landmarker | landmark facciali | Apache-2.0 | verificare il model card del bundle | registro pronto, download manuale |
| Real-ESRGAN | upscale neurale | BSD-3-Clause | verificare i termini della release prima di redistribuire | registro e download con consenso |
| 3DDFA_V2 | posa e geometria 3D | MIT | verificare separatamente i pesi | registro pronto, download manuale |
| InsightFace | controllo identità | MIT per il codice | molti pesi richiedono licenza separata | nessun auto-download |
| LaMa | rimozione oggetti non identitari | dipende dall'implementazione scelta | verificare peso e dataset | adattatore da aggiungere |
| CodeFormer | restauro generativo opzionale | verificare upstream | verificare peso e dipendenze | separato dalla modalità rigorosa |
| DFDNet | componenti facciali | verificare upstream | verificare peso e dataset | studio compatibilità |
| GFRNet | guida da riferimento | verificare upstream | verificare peso e dataset | studio compatibilità |

## Regole di integrazione

1. Nessun peso viene incorporato nell'installer senza una licenza che permetta esplicitamente la redistribuzione.
2. I download richiedono accettazione esplicita, URL HTTPS, limite di dimensione, scrittura atomica e checksum quando disponibile.
3. I modelli senza URL ufficiale stabile restano manuali.
4. InsightFace viene trattato come codice MIT con pesi soggetti a condizioni separate; non viene scaricato automaticamente.
5. I modelli generativi restano disattivati nella modalità conservativa rigorosa.
6. Se un modulo opzionale manca o fallisce, l'app deve continuare a funzionare e permettere di saltare il blocco.
7. Ogni risultato deve registrare modello, versione, licenza dichiarata e classificazione conservativa/generativa nel report di provenienza.

Il registro macchina è definito in `app/model_registry.py` e può essere esportato in JSON per l'interfaccia o per l'installer.
