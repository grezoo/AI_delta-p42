# KÖTELEZŐ MŰSZAKI ÉS VISELKEDÉSI SZABÁLYZAT (AGENTS.MD)

## 1. A LEÍRÁS AZ ELSŐ (SPECIFICATION FIRST MANDATE)
- Bármilyen kérdés, feladat vagy kétely merül fel, **MINDIG A MEGLÉVŐ KANONIKUS DOKUMENTÁCIÓ AZ ELSŐ (`megvalósítási terv.md`, `DELTA_P42_MUKODESI_ELVEK.md`, `Plan Review.md`, `DELTA_P42_MESTER_TEZIS.md`)!**
- **Kötelező sorrend:**
  1. Megnyitod a kanonikus tervfájlt, és megkeresed az adott fejezetet/táblázatot.
  2. **Ha a leírásban benne van:** Nem kérdezel, nem ajánlgatsz felesleges alternatívákat a megrendelőnek, hanem PONTOSAN AZT ÉS ÚGY ÉPÍTED MEG, ahogy a terv előírja!
  3. **Ha a leírásban nincs benne, vagy valódi ellentmondás van:** Idézed a tervet, rámutatsz a hiányra, és csak ekkor kérdezel célzottan a felhasználótól.
  4. **SZIGORÚAN TILOS a felhasználót arra kényszeríteni, hogy ő olvassa fel vagy keresse ki azt, amit a saját terve már rögzített!**

## 2. A „FAGYIS LOPÁS” ÉS KISHITŰ LEBUTÍTÁS SZIGORÚ TILALMA (ZERO MOCK MANDATE)
- Szigorúan tilos bármilyen neurális modellt, matematikai egyenletet vagy hardveres funkciót szintetikus sablonokkal, emulátorral vagy `force_simulation` ágakkal kiváltani!
- Ha a fizikai súlyfájl vagy a hardveres GPU allokáció hiányzik, a kódnak kötelezően `FileNotFoundError`-ral vagy `RuntimeError`-ral el kell buknia.
- Tilos hamis „PASS” minősítést kiadni olyan funkcióra, amely a valóságban nincs fizikailag letesztelve a hardveren.

## 3. ELŐBB VÁLASZOLSZ, UTÁNA PROGRAMOZS
- Bármilyen felhasználói észrevétel érkezik, ELŐSZÖR mindig szövegesen, emberi nyelven válaszolsz, elmagyarázva a kontextust és a lépéseket, és csak utána hajtasz végre műveletet.

## 4. KÓDHOZ NEM NYÚLSZ EGYEZTETÉS NÉLKÜL
- Meglévő működő logikához, release fájlokhoz vagy rögzített specifikációhoz tilos a felhasználó kifejezett jóváhagyása nélkül nyúlni.
