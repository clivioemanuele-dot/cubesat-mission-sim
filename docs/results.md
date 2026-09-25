# Risultati dello scenario di riferimento

> **Tutti i dati di questo documento sono simulati.** Non provengono da alcun
> satellite reale e non vanno usati per operazioni di volo.

Scenario: `config/baseline.toml`, seme 42, 24 ore a partire dal
2026-09-23 00:00 UTC. Le ipotesi e i limiti del modello sono in
`docs/assumptions.md`. Per riprodurre i numeri, dalla cartella del progetto:

    uv run python -m cubesat_sim.simulation config/baseline.toml results/baseline
    uv run python -m cubesat_sim.analysis results/baseline
    uv run python -m cubesat_sim.analysis.monte_carlo config/baseline.toml results/monte_carlo

La seconda riga stampa il riepilogo e salva `results/baseline/analysis.json`;
la terza esegue i 100 rilasci del Monte Carlo (§9). Per vedere tutto nella
dashboard:

    uv run python -m streamlit run src/cubesat_sim/dashboard/app.py

## 1. In sintesi

- Il satellite smorza la rotazione iniziale di 10 °/s in **1,73 h** con i soli
  magnetorquer, poi resta operativo per il resto della giornata: 10 cambi di
  modo, nessun SAFE, nessun ritorno in DETUMBLE.
- Quel rilascio è un po' fortunato: su 100 rilasci casuali il detumble dura in
  media 1,89 h e nel 95 % dei casi meno di **2,24 h** (§9). È questo il valore
  da usare in progetto.
- **L'energia non è il vincolo**: ogni orbita produce circa 20,6 Wh e ne
  consuma circa 3,2. La batteria resta sopra il 95 % dopo le prime due orbite.
- **Il vincolo sono i dati**: la fotocamera lavora solo 4 minuti al giorno
  (0,49 Gbit), per la geometria dei passaggi sull'Europa e per la priorità del
  DOWNLINK. Tutto quello che viene acquisito arriva a terra.
- **Il controllo d'assetto decide quanti dati scendono**: l'antenna punta la
  stazione entro metà fascio per il 72 % del tempo di visibilità. Il resto è
  la manovra iniziale verso Roma.
- Puntare la Terra invece del Sole costa **4,6 Wh per orbita (22 %)**.
- La batteria da 40 Wh è circa 16 volte la capacità minima che evita il SAFE
  (**2,45 Wh**): una scelta conservativa, giustificata dalla profondità di
  scarica e dai margini, non dal bilancio della giornata.

## 2. Detumble e modi

| Modo | Ore nelle 24 h | Note |
|---|---|---|
| DETUMBLE | 1,73 | Dal rilascio a 10 °/s fino a 0,5 °/s, con il B-dot (su 100 rilasci: §9) |
| SUN_POINTING | 21,91 | Modo predefinito |
| DOWNLINK | 0,29 | Tre passaggi su Roma: 10:00, 20:38 e 22:14 UTC (7,4, 7,1 e 2,6 min) |
| NADIR | 0,07 | Due finestre sull'Europa: 68 s e 176 s |
| SAFE | 0 | Mai attivato |

I due passaggi serali, sul ramo ascendente dell'orbita, avvengono in eclissi:
il trasmettitore lavora sulla batteria.

## 3. Puntamento

Errore vero dell'asse principale di ogni modo (pannelli verso il Sole,
fotocamera verso nadir, antenna verso la stazione):

| Modo | Medio | 95° percentile | Massimo |
|---|---|---|---|
| SUN_POINTING | 1,1° | 3,3° | 95° |
| NADIR | 5,8° | 25,6° | 29,6° |
| DOWNLINK | 16,2° | 68,3° | 110° |

- **SUN_POINTING**: per metà del tempo l'errore è sotto 0,3°. Il massimo di
  95° è l'inizio della manovra dopo il detumble; il 95° percentile risente
  delle manovre di rientro dopo ogni passaggio.
- **NADIR**: le finestre durano 1–3 minuti, mentre una manovra di 30° a 1 °/s
  con l'assestamento ne richiede circa uno e mezzo. La statistica è dominata
  dalla manovra: la fotocamera acquisisce anche mentre sta ancora ruotando
  (limite dichiarato).
- **DOWNLINK**: all'inizio di ogni passaggio l'antenna parte fino a 110° dalla
  stazione e la raggiunge in circa due minuti. Per questo il collegamento è
  disponibile per il 72 % della visibilità e non per il 100 %.

## 4. Ruote di reazione

Nessuna saturazione nelle 24 h. Il momento angolare immagazzinato cresce però
in modo quasi lineare, di circa **1,9 mN·m·s al giorno**: il dipolo residuo e
il gradiente gravitazionale non si annullano lungo l'orbita in un assetto
fisso rispetto al Sole. Alla fine della giornata la ruota più carica è al 22 %
della capacità. Senza lo scarico del momento con i magnetorquer (fuori dal
perimetro del progetto) una ruota arriverebbe al limite in circa 4–5 giorni:
per una missione reale lo scarico è indispensabile.

## 5. Energia

| Orbita | Prodotta | Consumata | Netta | Carica minima |
|---|---|---|---|---|
| 1 (detumble, parte in eclissi) | 6,9 Wh | 3,3 Wh | +3,6 Wh | 68,4 % |
| 2 | 20,5 Wh | 3,2 Wh | +17,4 Wh | 76,8 % |
| 3–15 | 19,7–20,7 Wh | 3,2–3,7 Wh | +16,0–17,5 Wh | 95,7–97,0 % |

- Al Sole, con i pannelli puntati, la potenza media è di 20,6 W; sull'orbita
  intera, eclissi compresa, 13,1 W. È il valore stimato a mano in
  `docs/assumptions.md` (§14), ora confermato.
- Le orbite con un passaggio su Roma consumano di più (fino a 3,7 Wh) per il
  trasmettitore da 4 W.
- La batteria è piena per gran parte del tempo: nelle 24 h vengono sprecati
  231 Wh su 296 prodotti. Il margine di potenza è molto ampio.

## 6. Dati

| Acquisiti | Scaricati | Rimasti a bordo | Persi |
|---|---|---|---|
| 0,49 Gbit | 0,49 Gbit | 0 | 0 |

La stima iniziale (circa 1,5 Gbit al giorno) era ottimistica. Con il nodo
discendente alle 10:30 solo due passaggi diurni attraversano il rettangolo
Europa, e il primo si sovrappone alla visibilità di Roma: il DOWNLINK, con
priorità più alta, interrompe il NADIR dopo 68 s. La memoria da 8 Gbit non si
avvicina mai al limite. Per acquisire di più servono scelte operative
(regione più ampia, priorità tra NADIR e DOWNLINK, pianificazione dei
passaggi), non più energia né più memoria.

## 7. Studio di compromesso 1: puntare la Terra o il Sole

Energia per orbita con l'assetto di riferimento ideale di ciascun modo
(`analysis/trade_studies.py`):

| | SUN_POINTING | NADIR |
|---|---|---|
| Potenza media al Sole | 20,6 W | 16,0 W |
| Energia per orbita | 20,5 Wh | 15,9 Wh |

Puntare la Terra costa **4,6 Wh per orbita, il 22 %**. In NADIR i pannelli +X
sono perpendicolari al nadir e ricevono il Sole con un fattore sin θ, con θ
angolo tra Sole e nadir: vicino al mezzogiorno orbitale θ ≈ 158° e il fattore
scende a 0,37.

Un minuto di NADIR al Sole costa circa 0,08 Wh di produzione persa, più
0,05 Wh per la fotocamera. Anche un'orbita intera in NADIR, con la fotocamera
accesa per tutto il tempo al Sole, chiuderebbe in attivo di quasi 10 Wh. Per
questo satellite l'energia non limita le acquisizioni.

## 8. Studio di compromesso 2: capacità della batteria

Finché il SAFE non scatta, la capacità della batteria non cambia il resto della
missione. Il modello della batteria è stato quindi rigiocato sui profili di
energia della simulazione, cercando per bisezione la capacità minima che tiene
la carica sopra il 30 %.

| Grandezza | Valore |
|---|---|
| Capacità minima senza SAFE | **2,45 Wh** |
| Prelievo massimo dalla batteria | 1,71 Wh (eclisse serale con DOWNLINK) |
| Profondità di scarica con 40 Wh | 4,3 % |

**Conferma con la simulazione completa** (24 h, stesso seme):

| Capacità | Risultato |
|---|---|
| 2,6 Wh | Nessun SAFE, carica minima 34 % |
| 2,3 Wh | SAFE alle 20,75 h, durante il passaggio serale in eclissi con il trasmettitore acceso; uscita dopo 4 minuti al Sole |

Il punto critico previsto dal rigioco è lo stesso della simulazione.

**Interpretazione.** Le batterie non si dimensionano sul minimo: in orbita
bassa si fanno circa 5500 cicli di carica e scarica l'anno, e per una vita di
diversi anni la regola pratica per le celle Li-ion è restare sotto il 20–30 %
di profondità di scarica per ciclo. Con 1,71 Wh per eclisse servono almeno
6–9 Wh. Vanno aggiunti i margini per ciò che qui non è simulato: rotazioni
lunghe dopo un'anomalia, invecchiamento, temperatura, crescita dei carichi.
Una batteria da 10–20 Wh sarebbe difendibile; i 40 Wh scelti sono
conservativi e corrispondono a pacchi commerciali comuni per i 3U.

## 9. Analisi Monte Carlo del detumble

Una simulazione descrive un solo rilascio. Per sapere entro quanto tempo il
detumble si chiude nella gran parte dei casi, lo scenario è stato ripetuto
con 100 semi diversi (da 1 a 100), simulando le prime 3 ore di ognuno
(`analysis/monte_carlo.py`). Tra una corsa e l'altra cambiano l'assetto e
l'asse di rotazione al rilascio e il rumore dei sensori; la velocità al
rilascio è sempre 10 °/s, il caso peggiore.

| Grandezza | Valore |
|---|---|
| Corse completate entro 3 h | 100 su 100 |
| Minimo | 0,83 h |
| Mediana | 1,94 h |
| Media | 1,89 h |
| **95° percentile** | **2,24 h** |
| Massimo | 2,33 h |

- Il seme 42 dello scenario di riferimento (1,73 h) sta sotto la mediana: il
  numero di una sola corsa era ottimistico di circa mezz'ora rispetto al
  valore di progetto.
- La distribuzione è asimmetrica: circa sette rilasci su dieci stanno tra 1,8
  e 2,3 h, con una coda di rilasci rapidi. Sono quelli in cui il satellite gira
  soprattutto attorno all'asse lungo Z, che ha un'inerzia sei volte minore: a
  parità di velocità angolare c'è meno momento angolare da smaltire con i
  magnetorquer. Il più rapido (seme 37, 0,83 h) parte con 9,2 °/s su Z.
- Conseguenza operativa: al 95 % il detumble occupa 1,4 orbite (periodo
  94,6 min). Le prime due orbite dopo il rilascio vanno pianificate senza
  acquisizioni né passaggi utili.

Le corse girano in parallelo su tutti i core del processore. I tempi di
detumble sono identici tra Windows e Linux. Limite dichiarato: non vengono
dispersi i parametri fisici (inerzia, dipolo residuo, velocità al rilascio,
guadagni del B-dot), che allargherebbero la distribuzione.

## 10. Problemi di sistema trovati dalla simulazione

Due difetti non emergevano dai test dei singoli componenti e sono comparsi
solo nella missione completa:

1. **Manovre grandi e DETUMBLE.** Con il PD classico una manovra di 180°
   arrivava a circa 8 °/s: sopra la soglia di 2 °/s, il gestore tornava in
   DETUMBLE, il B-dot non correggeva il puntamento e la manovra ricominciava.
   Soluzione: PD con limite di velocità a 1 °/s (`attitude/pointing.py`).
2. **Singolarità del riferimento DOWNLINK.** Nel passaggio serale la direzione
   di Roma passava a 10° da quella del Sole, usata per fissare la rotazione
   attorno all'antenna: il riferimento girava a 2,3 °/s e riportava il
   satellite in DETUMBLE. Soluzione: normale al piano dell'orbita come seconda
   direzione (`attitude/references.py`).

Sono il tipo di interazione tra controllo d'assetto e gestione dei modi che un
simulatore di missione serve a scoprire prima del volo.

## 11. Il simulatore

- **Riproducibilità**: con lo stesso seme due corse danno file identici byte
  per byte. Tra Windows e Linux i numeri differiscono solo all'ultima cifra
  (≤ 10⁻¹³ sull'assetto), con gli stessi eventi agli stessi secondi.
- **Tempo di calcolo**: 174 s per 24 h sul PC di sviluppo, dopo
  un'ottimizzazione guidata dal profilo dei tempi (§21 di
  `docs/assumptions.md`).
- **Verifiche**: 279 test automatici, dalla fisica dei singoli modelli alla
  missione completa e alla dashboard (§14 di `docs/assumptions.md`).
- **Dashboard** (`dashboard/app.py`, Streamlit): riproduce la giornata con un
  cursore del tempo (traccia a terra, assetto 3D, grafici di batteria,
  potenza, puntamento e ruote, linea temporale dei modi), mostra metriche,
  studi di compromesso e Monte Carlo, e rilancia simulazioni fino a 6 h con
  batteria, durata e rotazione al rilascio scelte dall'utente.

## 12. Possibili estensioni

- Scarico del momento delle ruote con i magnetorquer: necessario oltre i 4–5
  giorni.
- Pre-puntamento prima dell'inizio di un passaggio o di una finestra di
  acquisizione, per recuperare il 28 % di collegamento perso in manovra e i
  primi secondi di NADIR.
- Pianificazione delle acquisizioni e priorità tra NADIR e DOWNLINK.
- Puntamento al Sole ottimizzato: 21,8 W invece di 20,7 inclinando +X di circa
  18° verso ±Y.
- Monte Carlo esteso: dispersione dei parametri fisici e delle condizioni di
  rilascio, e statistiche anche su energia e dati della giornata intera.
- Stima dell'assetto (TRIAD, filtro di Kalman), orbita SGP4 da dati reali,
  modello magnetico IGRF completo, resistenza aerodinamica e pressione solare,
  link budget dettagliato.