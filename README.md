# TTTAI  
## Anforderungsanalyse

**Autor(en):**   Manuel  
**Erstellt am:** 12. Dezember 2025  
**Speicherort:** README.md  

---

## Inhalt

1. Install  
2. Einleitung  
   2.1 Systemidee  
   2.1.1 Die wichtigsten Funktionen  
   2.2 Management Summary  
3. Ausgangslage (IST)  
4. Ziele (SOLL)  
   4.1 Beschreibung der Ziele  
   4.2 Produktperspektive, Nutzen  
   4.3 Zielkonflikte  
   4.4 Abgrenzung  
5. Anforderungsanalyse  
   5.1 Identifizierung der Akteure  
   5.2 Anforderungskatalog  

---
## 1. Install

Clone the repo with:
```powershell
git clone https://github.com/The12Forest/TTTAI.git
```

Then cd into it with:
```powershell
cd TTTAI
```

Install the needed Packages with:
```powershell
npm install
```
if it dose not work use ``npm install express fs http https path socket.io ollama zod zod-to-json-schema``

Now finally start the Server with:
```powershell
node ./init.js
```

---
## 2. Einleitung

### 2.1 Systemidee

Das Grundprinzip ist es, dass es einen Server gibt, der eine Webseite hostet, welche es erlaubt, Tic Tac Toe zu spielen. Es soll zwischen Mensch oder AI gewählt werden können.  
Beim Spiel gegen einen Menschen kann ein anderer Spieler ausgewählt werden, mit dem anschliessend live gespielt wird. Zusätzlich soll es eventuell einen Live-Chat mit anderen Spielern geben.

Wird die AI gewählt, soll ein AI-Modell auf den Computer des Benutzers heruntergeladen werden. Danach kann gegen die AI gespielt werden, welche lokal auf dem Computer ausgeführt wird.

### 2.1.1 Die wichtigsten Funktionen

- Tic Tac Toe live spielen  
- Tic Tac Toe gegen eine AI spielen  
- Anmelden / Usermanagement  
- Live-Chat unter den Spielern  

### 2.2 Management Summary

*Noch offen*

---

## 3. Ausgangslage (IST)

Ich bin bereits gegen Ende des BLJ und habe keine Erfahrung mit der Implementierung von Neural Networks. Ziel ist es, eine Tic-Tac-Toe-AI zu erstellen.

---

## 4. Ziele (SOLL)

In diesem Kapitel werden die übergeordneten Ziele beschrieben, die mit dem zu entwickelnden System erreicht werden sollen.

### 4.1 Beschreibung der Ziele

Weitere Ziele sind:

- Ziel 1: Das System soll möglichst gut verhindern, dass im Spiel geschummelt wird (unter Berücksichtigung der Server-Performance).  
- Ziel 2: Der Nutzer soll Spass am Spielen haben.  
- Ziel 3: Es soll einen Live-Chat geben.  
- Ziel 4: Es soll möglich sein, in Echtzeit gegen einen echten Menschen zu spielen.  

### 4.2 Produktperspektive, Nutzen

Der Plan ist, dass dieses Produkt in der Freizeit genutzt wird, da es in der Arbeitswelt hauptsächlich für Arbeitszeitbetrug verwendet werden könnte.

### 4.3 Zielkonflikte

*Keine definiert*

### 4.4 Abgrenzung

Es besteht keine Möglichkeit, ein Chat-Monitoring umzusetzen, da dies zu viel zusätzliche Zeit benötigen würde und die nötige Leistung fehlt.

---

## 5. Anforderungsanalyse

Die Anforderungen an das zu entwickelnde System definieren alle zu erfüllenden Eigenschaften oder zu erbringenden Leistungen sowie allfällige technische Vorgaben.

### 5.1 Identifizierung der Akteure

- **Administrator:** Maximale Rechte, vollständige Systemkontrolle  
- **Spieler:** Benutzer mit Berechtigung zum Spielen und Chatten  

---

### 5.2 Anforderungskatalog

#### Funktionale Anforderungen

| Nr. | Akteur | Anforderung | Priorität | Erreicht |
|----|--------|-------------|-----------|---------|
| F-01 | Benutzer | Registrierung eines Benutzerkontos | 1 | Ja |
| F-02 | Benutzer | Anmeldung und Abmeldung | 1 | Ja |
| F-03 | Administrator | Verwaltung von Benutzerkonten (anzeigen, sperren, löschen) | 3 | Nein |
| F-04 | Spieler | Anzeige aller aktuell verfügbaren Spieler | 2 | Nein |
| F-05 | Spieler | Senden und Annehmen von Spielanfragen | 2 | Nicht dierekt |
| F-06 | Spieler | Durchführung eines Tic-Tac-Toe-Spiels in Echtzeit (Mensch vs. Mensch) | 1 | Ja |
| F-07 | Spieler | Durchführung eines Tic-Tac-Toe-Spiels gegen eine AI | 1 | Ja |
| F-08 | Spieler | Auswahl des Schwierigkeitsgrades der AI | 4 | Nein |
| F-09 | System | Serverseitige Prüfung aller Spielzüge | 1 | Nicht ganz |
| F-10 | Spieler | Korrekte Spielauswertung (Sieg, Niederlage, Unentschieden, Abbruch) | 1 | Ja |
| F-11 | Spieler | Nutzung eines Live-Chats während eines Spiels | 3 | Nein |
| F-12 | Administrator | Einsicht in System- und Spiel-Logs | 3 | In den Logs |

---

#### Offene Fragen

- [F1] Was genau ...?  
- [F2] Und wie …?  

#### Zusatzinformationen

- [Z1] none  

---

#### Nicht-funktionale Anforderungen

| Nr. | Anforderung | Priorität |
|----|-------------|-----------|
| NF-01 | Die Benutzeroberfläche muss übersichtlich und intuitiv bedienbar sein. | 2 |
| NF-02 | Das Spiel muss vollständig im Webbrowser spielbar sein. | 1 |
| NF-03 | Spielzüge müssen in Echtzeit verarbeitet werden (Antwortzeit < 1 Sekunde). | 1 |
| NF-04 | Das System muss mehrere parallele Spiele unterstützen. | 2 |
| NF-05 | Das System muss Manipulationsversuche erkennen und ungültige Spielzüge ablehnen. | 1 |
| NF-06 | Benutzersitzungen müssen sicher verwaltet werden. | 1 |
| NF-07 | Der Code muss modular und wartbar aufgebaut sein. | 3 |
| NF-08 | Die AI muss lokal auf dem Client ausgeführt werden. | 3 |
| NF-09 | Das System muss stabil und zuverlässig betrieben werden können. | 1 |
