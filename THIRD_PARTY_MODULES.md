# Moduli esterni e politica dei modelli

| Modulo production | Funzione | Licenza codice | Licenza pesi | Stato release |
|---|---|---|---|---|
| OpenCV / OpenCV Zoo YuNet | face detection e landmark 5 punti | MIT per la directory modello | MIT | ACTIVE, incluso offline |
| OpenCV / OpenCV Zoo SFace | identity/reference validation | Apache-2.0 | Apache-2.0 | ACTIVE, incluso offline |
| OpenCV / OpenCV Zoo NAFNet | deblur e denoise conservativo | MIT | MIT | ACTIVE, incluso offline |
| Face Parsing ResNet18 ONNX | semantic parsing | MIT | MIT upstream repository | ACTIVE, incluso offline |
| Head Pose MobileNetV2 ONNX | stima posa | MIT | MIT upstream repository | ACTIVE, incluso offline |
| OpenCV / OpenCV Zoo LaMa | residual non-evidence inpainting | Apache-2.0 | Apache-2.0 | FALLBACK, incluso offline |
| OpenCV runtime | immagini, DNN, geometria e fusion | Apache-2.0 | n/a | integrato |
| ONNX Runtime | inference CPU | MIT | n/a | integrato |
| PySide6 / Qt | interfaccia Windows | LGPLv3/GPLv3/commerciale secondo il pacchetto distribuito | n/a | integrato secondo i termini applicabili |

## Regole di integrazione

1. Nessun peso non verificato può essere ACTIVE/FALLBACK o entrare nell'installer.
2. I sei pesi production hanno URL ufficiale, dimensione massima, SHA-256, loader, inference e smoke test reali.
3. Gli aggiornamenti usano HTTPS, download temporaneo, SHA-256, smoke inference, attivazione atomica e rollback.
4. InsightFace resta DISABLED perché i pretrained pack upstream richiedono condizioni separate; non viene scaricato automaticamente.
5. CodeFormer, GFPGAN, RestoreFormer, DMDNet e gli altri modelli pesanti restano TESTING e non sono inclusi.
6. RefSTAR, InstantRestore e OSDFace restano OPTIONAL_RESEARCH con installazione production vietata finché licenza, pesi e runtime non sono verificati.
7. MODEL_INFERRED resta generato/low-confidence e contribuisce zero alla copertura di evidenza originale.

Il catalogo completo e machine-readable è incluso come `models/model-manifests.json`; lo stato runtime verificato è in `models/model-registry.json`.
