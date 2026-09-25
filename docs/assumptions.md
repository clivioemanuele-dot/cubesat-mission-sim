# Ipotesi e valori di riferimento

> **Tutti i dati prodotti da questo progetto sono simulati.** I valori sono
> plausibili per un CubeSat 3U reale, ma non descrivono un satellite specifico.

Questo documento giustifica ogni valore di `config/baseline.toml`, lo scenario
di riferimento, e dichiara i modelli e le approssimazioni del simulatore.
I criteri di scelta sono tre:

1. valori tipici di componenti commerciali per CubeSat;
2. coerenza fisica, verificata con i calcoli e i test della sezione 14;
3. tra più scelte ragionevoli, la più semplice.

Nel file TOML le unità sono indicate dal suffisso del nome (`_km`, `_deg`,
`_wh`...); il programma le converte in unità SI al caricamento.

## 1. Simulazione — `[simulation]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `seed` | 42 | Arbitrario ma esplicito. Con lo stesso seme e le stesse versioni dei pacchetti la simulazione è ripetibile bit per bit. Tra sistemi operativi diversi i numeri possono differire all'ultima cifra (≤ 10⁻¹³ sull'assetto, per le diverse librerie matematiche), con gli stessi eventi e gli stessi cambi di modo. |
| `start_epoch` | 2026-09-23 00:00 UTC | Giorno dell'equinozio: Sole sul piano equatoriale, illuminazione rappresentativa (angolo beta ≈ 22° per questa orbita). |
| `duration_h` | 24 | Circa 15 orbite: copre i passaggi sulla stazione di un giorno intero. In 24 h la precessione dell'orbita (~1°) è trascurabile. |
| `integration_step_s` | 0,25 | Alla velocità massima iniziale (10 °/s) il satellite ruota di 2,5° per passo: ben risolto da RK4. Il controllo a 1 Hz cade esattamente ogni 4 passi. |
| `output_step_s` | 10 | 8 640 righe per 24 h: sufficienti per i grafici, file leggeri. |

## 2. Satellite — `[satellite]`, `[initial_state]`

**Geometria e assi del corpo.**

- **Z** è l'asse lungo (34 cm). Sulla faccia **+Z** ci sono la fotocamera e
  l'antenna in banda S.
- Le due **ali dispiegabili** e le celle sulla faccia **+X** guardano verso +X.
- Gli assi del corpo coincidono con gli assi principali di inerzia.

Con +Z verso la Terra, l'asse di inerzia minima è radiale: è l'assetto
naturalmente stabile rispetto al gradiente gravitazionale (verificato in
`tests/test_disturbances.py`), quindi il puntamento a nadir richiede poco
sforzo di controllo.

| Parametro | Valore | Motivazione |
|---|---|---|
| `mass_kg` | 4,0 | Massa tipica di un 3U con ali dispiegabili, entro i limiti della CubeSat Design Specification. |
| `size_cm` | 10 × 10 × 34 | Formato standard 3U. |
| `inertia_kg_m2` | 0,042; 0,042; 0,007 | Parallelepipedo omogeneo: I_x = I_y = m(0,1² + 0,34²)/12 = 0,0419 kg·m²; I_z = m(0,1² + 0,1²)/12 = 0,0067 kg·m². Arrotondati. Le ali, sottili e leggere, sono trascurate. |
| `residual_dipole_am2` | (0; 0; 0,005) | Ordine di grandezza tipico dei CubeSat (1–10 mA·m²), dovuto a correnti di bordo non compensate. Coppia risultante ≈ 0,005 × 40 µT = 0,2 µN·m. |
| `max_rate_deg_s` | 10 | Limite superiore conservativo della rotazione dopo il rilascio dal deployer (tipicamente pochi °/s). La simulazione usa il caso peggiore: velocità pari al massimo, in direzione casuale; anche l'assetto iniziale è casuale. Entrambi vengono dal seme; 100 rilasci diversi sono nel Monte Carlo (sezione 22). |

## 3. Orbita — `[orbit]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `altitude_km` | 500 | Quota tipica dei lanci condivisi per CubeSat: vita orbitale di alcuni anni, resistenza aerodinamica moderata (trascurata in 24 h). |
| `inclination_deg` | 97,4 | Inclinazione eliosincrona a 500 km (sezione 14): il piano dell'orbita ruota di circa 1° al giorno, come il Sole. |
| `descending_node_local_time` | 10:30 | Orbita del mattino tipica dell'osservazione della Terra: illuminazione obliqua e poche nubi convettive, che si sviluppano nel pomeriggio. Il passaggio diurno è discendente (da nord a sud). |
| `initial_argument_of_latitude_deg` | 0 | Partenza sul nodo ascendente. Scelta convenzionale: la posizione al rilascio non è nota a priori. |

## 4. Stazione di terra — `[ground_station]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `name`, `latitude_deg`, `longitude_deg` | Roma, 41,90° N, 12,50° E | Stazione in Italia, come da requisito. |
| `altitude_m` | 50 | Quota tipica dell'area di Roma; effetto trascurabile sulla visibilità. |
| `min_elevation_deg` | 10 | Maschera di elevazione tipica: più in basso, ostacoli, cammini multipli e attenuazione atmosferica degradano il collegamento. Durata massima di un passaggio ≈ 7,4 min (sezione 14). |

## 5. Sensori — `[sensors]`

Deviazioni standard (1 sigma) per asse. **Non c'è filtro di stima**: il
controllore usa direttamente la misura rumorosa. È un limite dichiarato.

| Parametro | Valore | Motivazione |
|---|---|---|
| `attitude_noise_deg` | 0,02 | Ordine di grandezza di un piccolo star tracker per CubeSat (decine di secondi d'arco). |
| `rate_noise_deg_s` | 0,01 | Giroscopio MEMS campionato a 1 Hz. |
| `magnetometer_noise_nt` | 50 | Magnetometro tipico per CubeSat: circa lo 0,1–0,2 % del campo a 500 km (20–50 µT). |

## 6. Attuatori — `[actuators]`

Tre ruote di reazione e tre magnetorquer, ciascuno allineato con un asse del corpo.

| Parametro | Valore | Motivazione |
|---|---|---|
| `wheel_max_torque_mnm` | 1,0 | Classe delle ruote di reazione per CubeSat 1U–3U. |
| `wheel_max_momentum_mnms` | 10 | Stessa classe. All'uscita dal DETUMBLE (0,5 °/s) il satellite ha 0,042 × 0,0087 = 0,37 mN·m·s: meno del 4 % della capacità di una ruota. |
| `wheel_max_speed_rpm` | 6000 | Velocità tipica. Insieme al momento massimo fissa l'inerzia del rotore: J = 0,01 / 628,3 = 1,59·10⁻⁵ kg·m². |
| `magnetorquer_max_dipole_am2` | 0,2 | Classe dei magnetorquer per CubeSat 3U. Coppia massima ≈ 0,2 × 40 µT = 8 µN·m: sufficiente per il detumble, non per il puntamento fine, che richiede le ruote. |

## 7. Controllo — `[control]`

I guadagni non sono scritti a mano (sezione 19):

- **PD di puntamento**: per ogni asse K_p = I·ω_n² e K_d = 2·ζ·ω_n·I.
- **B-dot**: k = 2·n·(1 + sin i)·I_min = 3,09·10⁻⁵ N·m·s, ricavato da velocità
  orbitale, inclinazione e inerzia minima.

| Parametro | Valore | Motivazione |
|---|---|---|
| `control_period_s` | 1,0 | Frequenza tipica del software ADCS di un CubeSat. È circa 60 volte più veloce della banda del controllo, quindi gli effetti della discretizzazione sono piccoli. |
| `pd_natural_frequency_rad_s` | 0,1 | Assestamento al 2 % in circa 60 s (57 s con la formula approssimata 4/(ζ·ω_n)), verificato in `tests/test_pointing.py`. Vale per errori sotto circa 14°; oltre interviene il limite di velocità. |
| `pd_damping_ratio` | 0,7 | Compromesso classico tra rapidità e sovraelongazione (≈ 5 %). |
| `max_slew_rate_deg_s` | 1,0 | Velocità massima comandata nelle manovre di grande angolo (sezione 19). Senza limite, il PD porterebbe una manovra di 180° a circa 8 °/s: oltre la soglia del DETUMBLE (2 °/s) e con il 60 % della capacità delle ruote. Con 1 °/s una manovra di 180° dura circa 3 minuti e le ruote accumulano meno di 0,8 mN·m·s. La configurazione rifiuta valori non inferiori a `detumble_enter_rate_deg_s`. Nel DOWNLINK si somma il moto del riferimento (fino a 0,86 °/s): nella corsa di 24 h la velocità massima dopo il detumble è 1,57 °/s. |

## 8. Generatore solare — `[solar_array]`

Costante solare: **1361 W/m²** (valore nominale IAU 2015, Risoluzione B3;
sezione 17).

| Parametro | Valore | Motivazione |
|---|---|---|
| `cell_efficiency` | 0,30 | Celle GaAs a tripla giunzione: efficienza tipica di inizio vita a 28 °C, nello spettro solare fuori dall'atmosfera. |
| `derating_factor` | 0,80 | Prodotto delle perdite: temperatura del pannello ≈ 0,90 × conversione MPPT ≈ 0,95 × cablaggio e diodi ≈ 0,97 × vetro di copertura e invecchiamento ≈ 0,97 ≈ 0,80. |
| `panels` | 6 × 211,3 cm² | Ogni pannello porta 7 celle da 30,18 cm², un formato standard. Le due ali e la faccia +X guardano verso +X; le facce −X, +Y e −Y hanno celle montate sul corpo. Le facce ±Z non hanno celle: ospitano fotocamera, antenna e interfacce con il deployer. |
| `sun_pointing_axis_body` | +X | Normale comune alle ali e alla faccia +X: massimizza l'area esposta al Sole tra le direzioni degli assi. |

Con il Sole esattamente su +X la potenza è **20,7 W** a 1 UA. Il **massimo
geometrico** è però **21,8 W** (√10 volte un pannello), con il Sole a circa 18°
da +X verso ±Y: alle tre superfici su +X si aggiunge una faccia laterale.
È una possibile ottimizzazione del riferimento SUN_POINTING, elencata tra le
estensioni in `docs/results.md`.

## 9. Batteria — `[battery]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `capacity_wh` | 40 | Pacco tipico di un 3U: 4 celle Li-ion 18650 da circa 10 Wh ciascuna. |
| `charge_efficiency`, `discharge_efficiency` | 0,95 | Rendimento tipico delle celle Li-ion; il ciclo completo di carica e scarica vale circa 0,90. |
| `initial_soc_pct` | 70 | Carica parziale al rilascio: il satellite resta spento nel deployer anche per settimane. Ipotesi. |

## 10. Consumi — `[loads]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `obc_w` | 0,5 | Computer di bordo tipico per CubeSat, sempre acceso. |
| `adcs_base_w` | 1,5 | Sensori ed elettronica ADCS, compreso lo star tracker, sempre accesi (requisito: 1–2 W). |
| `wheel_max_power_w` | 0,5 | Per ruota, alla coppia massima; il consumo cresce in proporzione alla coppia comandata. |
| `magnetorquer_max_power_w` | 0,25 | Per magnetorquer, al dipolo massimo; il consumo cresce in proporzione al dipolo comandato. |
| `transmitter_w` | 4,0 | Trasmettitore in banda S (1–2 W a radiofrequenza), acceso per tutto il modo DOWNLINK. |
| `camera_w` | 3,0 | Fotocamera ed elettronica di elaborazione, accese per tutto il modo NADIR. |

## 11. Carico utile e dati — `[payload]`

| Parametro | Valore | Motivazione |
|---|---|---|
| `camera_boresight_body` | +Z | La fotocamera guarda lungo l'asse lungo, verso la Terra in NADIR. |
| `camera_data_rate_mbps` | 2 | Flusso di immagini compresse. La stima iniziale era un volume acquisito in un giorno (~1,5 Gbit) vicino a quello scaricabile (~1,2 Gbit). La prima corsa di 24 h la smentisce: la fotocamera acquisisce solo 0,49 Gbit, perché i passaggi diurni sull'Europa sono due e sopra l'Italia il DOWNLINK ha la priorità sul NADIR. Analisi in `docs/results.md` (§6). |
| `storage_capacity_gbit` | 8 | 1 GB, taglia tipica della memoria di massa di un CubeSat. È molto maggiore del volume giornaliero: nello scenario di riferimento la memoria non si riempie. |
| `antenna_boresight_body` | +Z | Sulla stessa faccia della fotocamera. |
| `antenna_beamwidth_deg` | 40 | Piccola schiera di antenne patch in banda S. I dati passano solo se l'errore di puntamento verso la stazione è entro metà fascio, cioè 20°. |
| `downlink_rate_mbps` | 1 | Requisito di progetto; plausibile in banda S con pochi watt a radiofrequenza da 500 km. Velocità costante: niente link budget dettagliato. |

## 12. Acquisizione immagini — `[imaging]`

La fotocamera acquisisce quando il punto sotto il satellite è nel rettangolo
**Europa** (latitudine 36°–60° N, longitudine 10° O–30° E) **e** il satellite
è illuminato dal Sole. La regione è vicina alla stazione: le immagini acquisite
vengono scaricate nei passaggi successivi su Roma. Che il satellite sia al Sole
viene usato come approssimazione del suolo illuminato.

## 13. Modi di bordo — `[modes]`

Priorità, dalla più alta: **SAFE**, **DETUMBLE**, **DOWNLINK**, **NADIR**,
**SUN_POINTING**. Ogni soglia ha un'isteresi, e ogni condizione deve durare
almeno `confirmation_time_s` prima del cambio di modo (`modes/manager.py`).

- Dopo il rilascio il gestore parte in **DETUMBLE**, come nella sequenza reale
  di un CubeSat.
- Decide solo con grandezze misurate o note a bordo: stato di carica,
  velocità angolare misurata, visibilità della stazione, finestra di
  acquisizione.
- Ogni transizione è registrata con istante, modo di partenza, modo di arrivo
  e causa (per esempio "batteria scarica (carica al 29.8 %)"), e finisce nei
  metadati dei risultati.
- In **SAFE**, se il satellite ruota ancora sopra la soglia del DETUMBLE, il
  controllo usa il B-dot invece del PD: puntare il Sole con le ruote durante
  una rotazione veloce le caricherebbe fino alla saturazione. La priorità di
  SAFE resta invariata.

| Parametro | Valore | Motivazione |
|---|---|---|
| `safe_enter_soc_pct` / `safe_exit_soc_pct` | 30 / 50 | Soglia di emergenza (requisito) con 20 punti di isteresi: 8 Wh da ricaricare, meno di un'orbita al Sole con i carichi ridotti. Evita ingressi e uscite continui. |
| `detumble_enter_rate_deg_s` | 2,0 | Oltre questa velocità le ruote accumulerebbero un momento eccessivo (1,5 mN·m·s sugli assi X e Y): si torna al B-dot. |
| `detumble_exit_rate_deg_s` | 0,5 | Il momento residuo (0,37 mN·m·s) è assorbito facilmente dalle ruote. Il rapporto 4:1 tra le soglie e il rumore del giroscopio (0,01 °/s) escludono oscillazioni tra i due modi. |
| `confirmation_time_s` | 10 | 10 cicli di controllo: filtra i picchi di rumore ed è trascurabile rispetto ai tempi dell'orbita. |

## 14. Verifiche di coerenza

Costanti della sezione 17. Semiasse maggiore a = R_E + 500 km = 6878,137 km.

**Orbita e ambiente**

| Grandezza | Formula | Valore | Verificata da |
|---|---|---|---|
| Periodo orbitale | T = 2π·√(a³/μ) | 94,6 min | `tests/test_orbit.py` |
| Velocità orbitale | v = √(μ/a) | 7,61 km/s | `tests/test_orbit.py` |
| Inclinazione eliosincrona | cos i = −2·ω☉·a^(7/2) / (3·J₂·R_E²·√μ) | 97,4° | `tests/test_orbit.py` |
| Ora locale al nodo discendente | dalla posizione del Sole all'epoca | 10:30 | `tests/test_orbit.py` |
| Angolo beta all'epoca di riferimento | elevazione del Sole sul piano orbitale | 22,3° | `tests/test_orbit.py` |
| Eclissi massima (beta = 0°) | T/π · arccos(√(a² − R_E²) / (a·cos β)) | 35,8 min | `tests/test_orbit.py`: confronto con l'ombra simulata per 4 valori di beta |
| Eclissi all'epoca di riferimento (beta ≈ 22°) | idem | 34,8 min | idem |
| Passaggio massimo sulla stazione (el ≥ 10°) | angolo al centro della Terra ≈ 14,0° per lato, diviso la velocità angolare orbitale | ≈ 7,4 min | `tests/test_ground.py`: 24 h simulate |
| Campo magnetico a 500 km | dipolo IGRF-14 | 24 µT (equatore) – 47 µT (poli) | `tests/test_magnetic.py` |

**Dinamica d'assetto e disturbi**

| Grandezza | Formula | Valore | Verificata da |
|---|---|---|---|
| Ordine dell'integratore RK4 | errore ∝ passo⁴ | errore / 16 dimezzando il passo | `tests/test_integrators.py` |
| Conservazione in rotazione libera (10 min a 10 °/s) | momento angolare in ECI ed energia costanti | deriva < 10⁻⁵ | `tests/test_integrators.py` |
| Scambio di momento con le ruote | h(t) = h₀ + τ·t, totale costante | esatto | `tests/test_integrators.py` |
| Instabilità dell'asse intermedio (effetto Dzhanibekov) | — | il corpo si capovolge | `tests/test_integrators.py` |
| Gradiente gravitazionale massimo | 3μ/(2a³)·(I_x − I_z) | 6,4·10⁻⁸ N·m | `tests/test_disturbances.py` |
| Stabilità dell'asse Z verso la Terra | coppia di richiamo | stabile (asse X: instabile) | `tests/test_disturbances.py` |
| Coppia del dipolo residuo | m × B | ≈ 0,2 µN·m | `tests/test_disturbances.py` |

**Sensori, attuatori e controllo**

| Grandezza | Criterio | Risultato atteso | Verificata da |
|---|---|---|---|
| Rumore dei sensori | deviazione standard su 2000 misure | come da configurazione, entro il 5 % | `tests/test_sensors.py` |
| Limiti delle ruote | 500 casi casuali | momento e coppia mai oltre i limiti | `tests/test_actuators.py` |
| Detumble con B-dot | mezz'orbita dal rilascio a 10 °/s, scenario completo | velocità sotto metà (senza controllo: sopra il 90 %) | `tests/test_bdot.py` |
| Risposta del PD a 10° di errore | assestamento al 2 % e sovraelongazione | 45–75 s (teoria ≈ 60 s); sotto il 10 % (teoria ≈ 5 %) | `tests/test_pointing.py` |
| Puntamento a nadir con rumore e disturbi | errore della fotocamera dopo l'assestamento | sotto 0,5°, nessuna saturazione | `tests/test_pointing.py` |
| Manovra di 170° con limite di velocità | velocità massima, saturazione, errore finale | sotto 1,05 °/s; nessuna saturazione; sotto 0,1° dopo 400 s | `tests/test_pointing.py` |
| Riferimento DOWNLINK nel passaggio serale su Roma | velocità del riferimento | sotto 1 °/s (con il Sole come seconda direzione: oltre 2 °/s) | `tests/test_references.py` |
| Gestore dei modi | priorità, isteresi, conferma di 10 s, registro delle cause | come da specifica | `tests/test_modes.py` |

**Energia e dati**

| Grandezza | Formula | Valore | Verificata da |
|---|---|---|---|
| Potenza di picco (Sole su +X, 1 UA) | 1361 × 3 × 0,02113 m² × 0,30 × 0,80 | 20,7 W | `tests/test_solar.py` |
| Potenza massima geometrica | √10 × un pannello | 21,8 W | `tests/test_solar.py` |
| Legge del coseno e distanza dal Sole | P ∝ cos θ, P ∝ 1/d² | esatte | `tests/test_solar.py` |
| Consumo di base | OBC + ADCS | 2,0 W | `tests/test_loads.py` |
| Consumo massimo (tutto acceso) | 2 + 1,5 + 0,75 + 4 + 3 | 11,25 W | `tests/test_loads.py` |
| Bilancio energetico | prodotta − consumata = Δbatteria + perdite + sprecata − non fornita | esatto | `tests/test_battery.py` |
| Conservazione dei dati | memoria = iniziale + acquisiti − scaricati | esatta | `tests/test_data.py` |
| Potenza media per orbita al Sole | 20,7 W × (1 − 34,8/94,6) | ≈ 13,1 W | corsa di 24 h, `docs/results.md` (§5) |

Il bilancio energetico è ampiamente positivo: anche con tutto acceso il
consumo (11,25 W) è circa metà della potenza di picco. Nello scenario di
riferimento il modo SAFE non è atteso. Lo studio di compromesso 2
(`docs/results.md`, §8) trova la capacità minima che lo evita: 2,45 Wh.

**Simulazione completa**

| Grandezza | Criterio | Risultato atteso | Verificata da |
|---|---|---|---|
| Riproducibilità | due corse con lo stesso seme; una con seme diverso | identiche bit per bit; diversa | `tests/test_simulation.py` |
| Bilancio energetico sull'intera corsa | prodotta − consumata = Δbatteria + perdite + sprecata − non fornita | esatto (10⁻⁹ relativo) | `tests/test_simulation.py` |
| Rilascio quasi fermo sul lato illuminato | transizioni, velocità, errore e potenza finali | una sola transizione; sotto 2 °/s; sotto 1°; oltre 19 W | `tests/test_simulation.py` |
| SAFE con rotazione veloce | controllo e carichi | B-dot, ruote ferme, solo piattaforma e magnetorquer accesi | `tests/test_simulation.py` |
| Salvataggio e rilettura | CSV e JSON | contenuto identico; file ripetibili byte per byte, fine riga LF | `tests/test_results.py` |
| Riga di comando | corsa breve salvata e riletta | 18 righe per 3 minuti, avviso "DATI SIMULATI" | `tests/test_main.py` |
| Tempo di detumble con il SAFE di mezzo | sequenze di modi costruite a mano | fine al primo modo di puntamento, non all'ingresso in SAFE | `tests/test_metrics.py` |
| Monte Carlo in parallelo | stesse corse in 2 processi e una per volta | identiche, nell'ordine dei semi; corse non finite contate a parte | `tests/test_monte_carlo.py` |
| Dashboard | pagina eseguita senza browser (AppTest di Streamlit) | avviso "SIMULAZIONE", campione giusto per ogni istante, frecce 3D coincidenti con i riferimenti dei modi | `tests/test_dashboard.py` |

**Scenario di riferimento, 24 h** (analisi completa in `docs/results.md`):

| Grandezza | Valore |
|---|---|
| Tempo di detumble (da 10 °/s a 0,5 °/s) | 1,73 h |
| Tempo di detumble su 100 rilasci (Monte Carlo) | 95 % entro 2,24 h; massimo 2,33 h |
| Cambi di modo | 10, nessun ritorno in DETUMBLE |
| Velocità angolare massima dopo il detumble | 1,57 °/s |
| Carica minima della batteria | 68 % |
| Saturazione delle ruote | nessuna; momento massimo 2,2 mN·m·s |
| Dati acquisiti e scaricati | 0,49 Gbit, scaricati tutti |

## 15. Limiti del modello (dichiarati)

**Orbita e ambiente**

- Orbita circolare kepleriana, senza perturbazioni: niente SGP4 né dati reali.
- Piano orbitale fisso in ECI: senza la precessione dovuta a J₂, in 24 h l'ora
  locale del nodo si sposta di circa 4 minuti.
- Sole da formula analitica a bassa precisione (≈ 0,01°). La direzione del Sole
  è presa dal centro della Terra, non dal satellite (differenza ≈ 0,003°).
  L'ora locale del nodo è riferita al Sole vero, non al Sole medio.
- Eclissi con ombra cilindrica, senza penombra (≈ 10 s a ogni ingresso e uscita).
- Terra sferica, con il raggio equatoriale, per l'ombra e per il punto sotto il
  satellite (errore di latitudine ≤ 0,2°). La stazione usa invece l'ellissoide
  WGS-84.
- Precessione e nutazione dell'asse terrestre trascurate (errore < 0,5° fino
  al 2030); UTC, UT1 e TT considerati coincidenti (differenze < 70 s).
- Campo magnetico terrestre come dipolo inclinato IGRF-14 (epoca 2025), non il
  modello completo: errore locale dell'ordine del 10 %, maggiore sopra
  l'Anomalia del Sud Atlantico.
- Resistenza aerodinamica e pressione solare trascurate.

**Dinamica d'assetto**

- Corpo rigido: ali rigide, inerzia diagonale e costante.
- Ruote allineate con gli assi principali; trascurato l'accoppiamento dovuto
  all'inerzia propria delle ruote (1,6·10⁻⁵ kg·m², contro 0,007–0,042 kg·m²
  del corpo).
- Comandi agli attuatori costanti durante ogni periodo di controllo (1 s);
  posizione e campo magnetico costanti durante lo stesso periodo.
- Nessuno scarico del momento delle ruote: i disturbi che non si annullano
  lungo l'orbita si accumulano nelle ruote (metrica di saturazione, step 8).

**Sensori e controllo**

- Assetto noto con rumore di misura, senza filtro di stima (niente TRIAD né
  filtro di Kalman). Rumore gaussiano senza bias.
- Il campo prodotto dai magnetorquer non disturba il magnetometro.
- Saturazioni: il vettore richiesto è ridotto in proporzione, conservando la
  direzione.
- Guadagno del B-dot calcolato con l'inclinazione dell'orbita al posto di
  quella rispetto all'equatore geomagnetico.
- Velocità del riferimento stimata da due riferimenti a 1 s di distanza;
  accelerazione del riferimento e termini giroscopici non compensati (effetto
  trascurabile sotto 1 °/s).
- Manovre di grande angolo a velocità limitata attorno all'asse dell'errore,
  senza profili di manovra ottimizzati.
- Conoscenze di bordo considerate esatte: posizione sull'orbita e ora (quindi
  eclissi e visibilità della stazione), stato di carica della batteria,
  momento delle ruote (letto dai tachimetri).

**Energia e dati**

- Pannelli: nessuna ombra reciproca tra ali e corpo, nessuna luce riflessa
  dalla Terra (albedo), nessuna perdita aggiuntiva con il Sole radente;
  efficienza costante, con la temperatura inclusa in media nel derating.
- Batteria come serbatoio di energia con rendimenti costanti: nessun modello
  di tensione, temperatura o invecchiamento. A batteria piena l'eccesso di
  produzione è sprecato.
- Consumo delle ruote proporzionale alla coppia, senza dipendenza dalla
  velocità di rotazione e senza attriti.
- Scarico dati "tutto o niente", a velocità costante se il puntamento è entro
  metà fascio: nessun link budget dettagliato.
- Illuminazione del suolo approssimata con quella del satellite.
- Fotocamera accesa per tutto il NADIR e trasmettitore per tutto il DOWNLINK,
  comprese le manovre iniziali: le immagini acquisite durante la manovra
  verso nadir sono contate come valide.
- Potenze costanti in ogni periodo di controllo (1 s), calcolate con
  l'assetto all'inizio del periodo.
- Nessun modello termico.

**Analisi**

- Tempo di detumble misurato al primo ingresso in un modo di puntamento: se il
  SAFE scatta durante il detumble, comprende l'attesa in SAFE ed è quindi un
  limite superiore.
- Monte Carlo sul solo seme: variano assetto e asse di rotazione al rilascio e
  rumore dei sensori; inerzia, dipolo residuo, velocità al rilascio e guadagni
  restano quelli nominali.

## 16. Sistemi di riferimento e tempo

| Sistema | Origine | Assi | Usato per |
|---|---|---|---|
| **ECI** (inerziale) | Centro della Terra | z verso il polo nord, x verso l'equinozio di J2000 | Orbita, Sole, dinamica d'assetto |
| **ECEF** (terrestre) | Centro della Terra | z verso il polo nord, x verso il meridiano di Greenwich; ruota rispetto a ECI attorno a z con l'angolo di rotazione terrestre (IERS 2010) | Stazione di terra, campo magnetico, traccia a terra |
| **LVLH** (orbitale) | Satellite | z verso il centro della Terra (nadir), y opposto al momento angolare dell'orbita, x lungo la velocità | Riferimento per il puntamento a nadir |
| **Corpo** | Baricentro del satellite | Assi principali di inerzia: Z lungo l'asse lungo, +Z verso fotocamera e antenna, +X verso le ali | Assetto, pannelli, attuatori |

**Matrici di rotazione** (`core/rotations.py`): "passive", cambiano il sistema
in cui un vettore è espresso senza muoverlo (`v_B = R @ v_A`).

**Quaternioni** (`core/quaternions.py`):

- ordine [w, x, y, z], con la parte scalare per prima; prodotto di Hamilton;
- il quaternione di assetto q descrive il corpo rispetto a ECI:
  `v_eci = R(q) @ v_body`;
- le rotazioni si compongono come i pedici: q_ac = q_ab ⊗ q_bc;
- cinematica: dq/dt = ½ · q ⊗ [0, ω], con ω espressa negli assi del corpo;
- q e −q rappresentano la stessa rotazione.

Per una rotazione di un angolo θ attorno a un asse, R(q) = rot(θ)ᵀ: la matrice
del quaternione è attiva, quelle di `rotations.py` sono passive.

**Tempo.** Il simulatore conta i secondi dall'epoca iniziale (UTC). Per il Sole
e la rotazione terrestre il tempo è convertito in data giuliana.

## 17. Costanti fisiche

Definite in `src/cubesat_sim/core/constants.py`, salvo dove indicato.

| Costante | Valore | Fonte |
|---|---|---|
| Parametro gravitazionale terrestre μ | 3,986004418·10¹⁴ m³/s² | WGS-84 |
| Raggio equatoriale R_E | 6 378 137 m | WGS-84 |
| Schiacciamento f | 1/298,257223563 | WGS-84 |
| Velocità di rotazione terrestre | 7,292115·10⁻⁵ rad/s | WGS-84 |
| J₂ (solo per le verifiche) | 1,08262668·10⁻³ | EGM2008 |
| Costante solare | 1361 W/m² | IAU 2015, Risoluzione B3 |
| Unità astronomica | 1,495978707·10¹¹ m | IAU 2012, Risoluzione B2 |
| Anno tropico | 365,2422 giorni | valore astronomico standard |
| Coefficienti di dipolo g₁⁰, g₁¹, h₁¹ | −29 350, −1 410, 4 545 nT | IGRF-14, epoca 2025, arrotondati al nT |
| Raggio di riferimento IGRF | 6 371,2 km | IGRF (in `environment/magnetic.py`) |
| Angolo di rotazione terrestre | 0,7790572732640 + 1,00273781191135448 giri al giorno da J2000 | IERS Conventions 2010 (in `environment/earth.py`) |
| Posizione del Sole | formula a bassa precisione | Astronomical Almanac, sezione C (in `environment/sun.py`) |

## 18. Modello di assetto

**Stato** (10 numeri, in assi corpo): quaternione di assetto q, velocità
angolare ω [rad/s], momento angolare delle ruote h [N·m·s].

**Equazioni di Eulero con le ruote** (`attitude/dynamics.py`):

    I · dω/dt = τ_esterna − τ_ruote − ω × (I·ω + h)
    dh/dt     = τ_ruote

τ_ruote è la coppia che i motori applicano alle ruote: sul corpo agisce la
reazione −τ_ruote. Il momento angolare totale I·ω + h cambia solo per effetto
delle coppie esterne.

**Integrazione** (`core/integrators.py`): RK4 a passo fisso di 0,25 s. Le
coppie esterne sono ricalcolate a ogni stadio, perché dipendono dall'assetto.
Dopo ogni passo il quaternione è rinormalizzato a norma 1.

**Disturbi** (`attitude/disturbances.py`):

| Coppia | Formula | Ordine di grandezza |
|---|---|---|
| Gradiente gravitazionale | τ = 3μ/r³ · r̂ × (I·r̂) | ≤ 0,064 µN·m |
| Dipolo magnetico residuo | τ = m × B | ≈ 0,2 µN·m |

Entrambe sono circa 10 000 volte più piccole della coppia massima di una ruota
(1 mN·m), ma agiscono sempre: la parte che non si annulla lungo l'orbita si
accumula nelle ruote.

## 19. Sensori, attuatori e controllo

**Separazione tra stato vero e misurato.** I controllori ricevono solo le
misure dei sensori (`attitude/sensors.py`), mai lo stato vero della
simulazione. Le coppie fisiche usano invece lo stato e il campo veri.

**Sensori**: star tracker (assetto), giroscopio (velocità angolare),
magnetometro (campo nel corpo), con rumore gaussiano per asse (sezione 5)
generato da un seme esplicito.

**Attuatori** (`attitude/actuators.py`):

- ruote: coppia massima 1 mN·m e momento massimo 10 mN·m·s per ruota. La coppia
  è limitata in modo che, mantenuta per un periodo di controllo, nessuna ruota
  superi il momento massimo. Ogni limite raggiunto è registrato come evento di
  saturazione;
- magnetorquer: dipolo massimo 0,2 A·m² per asse; coppia m × B.

**B-dot** (`attitude/bdot.py`), usato in DETUMBLE:

    m = −(k / |B|) · db/dt,      b = B/|B|,      k = 2·n·(1 + sin i)·I_min

La derivata è stimata da due misure consecutive del magnetometro. La coppia
risultante vale circa −k·ω⊥ e frena la rotazione. La velocità residua tende a
circa due volte la velocità orbitale (≈ 0,13 °/s), sotto la soglia di uscita
dal DETUMBLE.

**Assetti di riferimento** (`attitude/references.py`): un asse del corpo
allineato esattamente, un secondo il più vicino possibile a una seconda
direzione.

| Modo | Asse primario (esatto) | Asse secondario (il più vicino possibile) |
|---|---|---|
| SUN_POINTING, SAFE | +X (pannelli) → Sole | +Z (fotocamera) → Terra |
| NADIR | +Z (fotocamera) → centro della Terra | +X (pannelli) → Sole |
| DOWNLINK | +Z (antenna) → stazione | +X (pannelli) → normale al piano dell'orbita, dalla parte del Sole |

La velocità del riferimento è stimata da due riferimenti a 1 s di distanza e
fornita al controllore: senza, l'inseguimento della stazione avrebbe un
ritardo di circa 12°.

**Singolarità del riferimento.** Quando la direzione primaria e la secondaria
diventano quasi parallele, la rotazione attorno all'asse primario è mal
definita e il riferimento ruota velocemente. Con il Sole come seconda
direzione del DOWNLINK succedeva davvero. Nel passaggio serale su Roma, in
eclissi, la direzione della stazione passa a 10° da quella del Sole: il
riferimento arrivava a 2,3 °/s e riportava il satellite in DETUMBLE. Con la
normale al piano dell'orbita non succede più: la stazione si vede sempre entro
circa 64° dal nadir, quindi ad almeno 26° dalla normale, e il riferimento resta
sotto 0,86 °/s. La rotazione attorno all'antenna non conta per il
collegamento. Il costo è piccolo: nei passaggi diurni i pannelli vedono il Sole
a 90° − |β| ≈ 68° (circa 7,7 W invece di 20). In SUN_POINTING e NADIR, Sole e
nadir restano ad almeno |β| ≈ 22° di distanza e il riferimento non supera
0,2 °/s.

**PD di puntamento con limite di velocità** (`attitude/pointing.py`), usato
negli altri modi:

    ω_cmd = −(ω_n / 2ζ)·φ,  ridotta in modulo a ω_max
    τ = −K_d·(ω − ω_rif − ω_cmd),    K_p = I·ω_n²,    K_d = 2·ζ·ω_n·I

φ è il vettore di rotazione dell'errore d'assetto (modulo = angolo d'errore).
Finché |ω_cmd| < ω_max, cioè per errori sotto circa 14°, la legge coincide con
il PD classico τ = −K_p·φ − K_d·(ω − ω_rif). Per errori più grandi il satellite
ruota a velocità costante ω_max = 1 °/s attorno all'asse dell'errore, il
percorso più breve, e poi si assesta come un PD. Senza il limite, ogni grande
manovra supererebbe la soglia del DETUMBLE: il gestore tornerebbe al B-dot, che
non corregge il puntamento, e la manovra ricomincerebbe da capo.

L'**errore di puntamento** di uno strumento è l'angolo tra il suo asse e il
bersaglio, ed è la metrica usata allo step 8.

## 20. Energia e dati

Il sottopacchetto `resources` dipende solo da `core`: riceve dalla simulazione
direzione del Sole negli assi del corpo, eclissi, comandi degli attuatori,
carichi accesi, visibilità della stazione ed errore dell'antenna.

**Pannelli** (`resources/solar.py`):

    P = S·(UA/d)² · η · derating · Σ Aᵢ · max(0, nᵢ · ŝ),    P = 0 in eclissi

**Batteria** (`resources/battery.py`): carica con rendimento 0,95, scarica con
rendimento 0,95, stato di carica tra 0 e 1. A ogni aggiornamento:

    prodotta − consumata = Δbatteria + perdite + sprecata − non fornita

**Consumi** (`resources/loads.py`):

| Sottosistema | Consumo |
|---|---|
| Computer di bordo + ADCS | 2,0 W, sempre |
| Ruote | 0,5 W × \|coppia\| / coppia massima, per ruota |
| Magnetorquer | 0,25 W × \|dipolo\| / dipolo massimo, per asse |
| Trasmettitore | 4 W, per tutto il DOWNLINK |
| Fotocamera | 3 W, per tutto il NADIR |

Quali carichi sono accesi lo decide il gestore dei modi (in SAFE trasmettitore
e fotocamera sono spenti).

**Memoria dati** (`resources/data.py`): acquisizione a 2 Mbit/s con la
fotocamera accesa; scarico a 1 Mbit/s solo se la stazione è visibile **e**
l'antenna punta entro 20° da essa. A memoria piena i dati in più sono persi. I
dati si conservano: memoria = iniziale + acquisiti − scaricati.

## 21. Ciclo di simulazione e risultati

**Ciclo** (`simulation/runner.py`), a ogni periodo di controllo di 1 s:

1. ambiente all'istante t: orbita, Sole, eclissi, campo magnetico, stazione,
   punto sotto il satellite, finestra di acquisizione;
2. misure dei sensori sullo stato vero;
3. gestore dei modi;
4. comandi, calcolati solo dalle misure: B-dot con i magnetorquer in DETUMBLE
   (e in SAFE con rotazione veloce), altrimenti PD con le ruote verso il
   riferimento del modo;
5. energia e dati: pannelli e collegamento radio con l'assetto vero, consumi
   dai comandi, batteria e memoria;
6. dinamica vera: 4 passi di RK4 da 0,25 s con comandi costanti.

**Risultati** (`core/results.py`): una cartella con due file, entrambi con
l'avviso "DATI SIMULATI".

- `timeseries.csv`, una riga ogni 10 s, unità SI indicate dal suffisso della
  colonna:
  - tempo e modo;
  - latitudine e longitudine del punto sotto il satellite;
  - eclissi, visibilità della stazione, finestra di acquisizione;
  - quaternione e velocità angolare veri, momento delle ruote;
  - errore di puntamento del modo ed errore dell'antenna;
  - potenza prodotta e consumata, stato di carica;
  - contatori cumulati di energia, dati e tempo di saturazione delle ruote.

  Ogni riga descrive l'istante t; i contatori sommano ciò che è avvenuto prima
  di t, così il bilancio energetico si chiude su ogni riga.
- `metadata.json`: seme, configurazione completa (basta per rilanciare la
  stessa simulazione), condizioni iniziali, transizioni di modo con la causa.

**Esecuzione**, dalla cartella del progetto:

    uv run python -m cubesat_sim.simulation config/baseline.toml results/baseline

L'opzione `--hours` cambia la durata per prove brevi.

**Prestazioni.** Il profilo dei tempi (`cProfile`) ha mostrato che `np.cross`
prendeva il 67 % del tempo di calcolo. È una funzione generale per array di
vettori, e per un singolo vettore 3D spende quasi tutto il tempo in controlli.
Il prodotto vettoriale e la norma scritti per vettori 3D (`core/vectors.py`),
più l'estrazione delle componenti dei quaternioni con `tolist()`, danno
risultati identici bit per bit. Sul PC di sviluppo la corsa di 24 h è scesa da
451 s a 174 s. Il profilo che resta è piatto: il tempo è distribuito sulle
circa 16 valutazioni delle equazioni del moto per secondo simulato. Andare
oltre richiederebbe di riscrivere la dinamica in forma meno leggibile o una
dipendenza in più, fuori dal perimetro del progetto.

## 22. Monte Carlo e dashboard

**Monte Carlo del detumble** (`analysis/monte_carlo.py`). Lo scenario di
riferimento è ripetuto con i semi da 1 a 100, 3 ore simulate per corsa: su
100 rilasci il detumble più lungo è di 2,3 h, quindi la finestra lascia
margine; le corse che non finiscono entro la finestra sono contate a parte e
non entrano nelle statistiche. Le corse girano in parallelo su più processi
(`concurrent.futures` della libreria standard); ognuna è indipendente e il
risultato non dipende dal numero di processi. Risultati in
`results/monte_carlo`: `detumble_runs.csv` (una riga per seme) e
`summary.json` (parametri e statistica), entrambi con l'avviso "DATI
SIMULATI".

    uv run python -m cubesat_sim.analysis.monte_carlo config/baseline.toml results/monte_carlo

**Dashboard** (`dashboard/app.py` e `dashboard/figures.py`). Sola
visualizzazione: legge i risultati salvati e ricalcola le metriche con le
stesse funzioni dell'analisi. Le figure sono funzioni pure, verificate senza
browser. La vista 3D ricalcola le direzioni di Sole, Terra e stazione con gli
stessi modelli analitici della simulazione. Una nuova simulazione dalla barra
laterale parte dallo scenario di riferimento, con il suo seme, e dura al
massimo 6 ore (circa 45 s di calcolo sul PC di sviluppo); resta in memoria e
non viene salvata.

    uv run python -m streamlit run src/cubesat_sim/dashboard/app.py