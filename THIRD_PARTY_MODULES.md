# Moduli esterni pianificati

| Modulo | Funzione | Integrazione prevista | Stato |
|---|---|---|---|
| OpenCV | deblur, denoise, colore, maschere, blending | integrato nel core | attivo |
| Real-ESRGAN NCNN | upscale x2/x4 | processo esterno Vulkan | rilevamento pronto |
| LaMa | rimozione oggetti | ONNX Runtime CPU | adattatore da aggiungere |
| 3DDFA_V2 | posa e frontalizzazione | ONNX Runtime CPU | adattatore da aggiungere |
| InsightFace | allineamento e verifica identità | ONNX Runtime CPU | adattatore da aggiungere |
| CodeFormer | restauro generativo opzionale | processo separato | adattatore da aggiungere |
| DFDNet | componenti facciali | ambiente opzionale | studio compatibilità |
| GFRNet | guida da riferimento | riscrittura/adattamento | studio compatibilità |

Principio del progetto: ogni modulo opzionale è isolato. Se manca o fallisce, l'app continua a funzionare e permette di saltare il blocco.
