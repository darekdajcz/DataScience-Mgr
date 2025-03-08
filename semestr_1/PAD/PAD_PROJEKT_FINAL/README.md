# Dokumentacja projektu - Analiza COVID-19

## Opis struktury projektu
Projekt zawiera analizę danych dotyczących COVID-19, obejmującą czyszczenie danych, analizę oraz wizualizację wyników.

Lista plików projektu:

* covid_analysis7.ipynb - Główny plik projektu zawierający kod analityczny w formacie Jupyter Notebook.

// Folder data - Zawiera pliki z danymi wykorzystywanymi w projekcie:
* daily-covid-19-vaccine-doses-administered.csv - Zbiór danych dotyczący dziennej liczby podawanych dawek szczepionek przeciwko COVID-19.
* daily_covid_info.csv - Plik zawierający dodatkowe informacje dotyczące pandemii COVID-19.

## Struktura projektu

Czyszczenie danych - Wczytanie i przetworzenie danych (np. usunięcie brakujących wartości, formatowanie kolumn).

Analiza danych - Przeprowadzenie analiz statystycznych i eksploracyjnych (np. trendy).

Modelowanie - Ewentualne predykcje na podstawie dostępnych danych.

Dashboard / Wnioski - Prezentacja wyników predykcji w czytelnej formie oraz wnioski.

# Instrukcja uruchomienia

## Wymagania systemowe

Aby uruchomić projekt, wymagane jest:

* Python w wersji 3.8 lub nowszej

* Jupyter Notebook lub inny kompatybilny środowisko do uruchamiania plików .ipynb

Instalacja wymaganych bibliotek

Przed uruchomieniem należy zainstalować wymagane biblioteki. Można to zrobić za pomocą pliku requirements.txt (jeśli jest dostępny) lub ręcznie:

``pip install pandas matplotlib seaborn numpy`` 

Uruchomienie analizy

Pobierz pliki projektu i umieść je w jednym katalogu.

Otwórz terminal i przejdź do katalogu projektu.
Uruchom Jupyter Notebook:
jupyter notebook
Otwórz plik covid_analysis7.ipynb i wykonaj poszczególne komórki kodu