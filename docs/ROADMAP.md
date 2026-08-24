# Roadmap tecnica

## V1 — pipeline locale a blocchi

- [x] Importazione immagine principale
- [x] Deblur/denoise conservativo
- [x] Contrasto locale
- [x] Accetta, riprova, salta e continua
- [x] Esportazione PNG/JPEG/TIFF
- [ ] Salvataggio dei risultati intermedi
- [ ] Importazione multipla delle reference

## V2 — moduli esterni

- [ ] Plugin Real-ESRGAN NCNN/Vulkan
- [ ] Plugin LaMa per inpainting
- [ ] Face detection e landmark
- [ ] Rilevamento automatico delle occlusioni

## V3 — ricostruzione multi-foto

- [ ] Allineamento delle reference
- [ ] Selezione e fusione regionale
- [ ] Frontalizzazione 3D
- [ ] Controllo identitario
- [ ] Provenienza delle regioni

## Vincoli

- CPU-first
- obiettivo inferiore a cinque minuti per foto
- elaborazione locale
- nessun modello generativo obbligatorio
- ogni blocco deve poter essere accettato, ripetuto o saltato
