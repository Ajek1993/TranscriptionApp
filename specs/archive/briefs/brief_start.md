Hej, słuchaj więc. Chcę stworzyć prostą aplikację CLI, która rozwiąże mój problem, i w kolejnych krokach będę ci ten mój problem opisywał.

Q: Co dokładnie boli?
A: Nagrywam sporo materiałów wideo na podstawie których później generuje transkrypty a następnie przygotowuje materiały w postaci prezentacji, ale nie mam jak zrobić odnośników do danego momentu w danych filmie wideo, aby wskazać źródła jakby. Potrzebuję dokładnie timestampy

Q: Dla kogo to rozwiązanie?
A: To rozwiązanie jest tylko dla mnie, dla autora tych materiałów wideo.

Q: Dlaczego CLI jest najlepsze?
A: Chcę, żeby to była aplikacja CLA, ponieważ chcę, żeby była dostępna z konsoli, bo mi się wygodniej w konsoli pisze, to jest raz. A dwa. Chcesz, żeby była łatwo dostępna dla agentów AI i dla modeli, z którymi pracuje. Dlatego chcesz, żeby to było z jesteście, bo one będą miały łatwy dostęp do tej aplikacji.

Q: Jakie są główne potrzeby?
A: Przy pomocy najlepszego obecnie modelu do transkryptów nvidia parakeet v3, zbudować cli, które zrobi transkrypt materiału video (najlepiej z możliwością podania linku do filmu na yt), i zapisze go z timestampami na dysku. To jest taka rzecz, która bardzo by mi ułatwiła życie. Muszę użyć narzędzie z linku: źrodło: https://www.digitalocean.com/community/tutorials/srt-generation-parakeet-autocaption

Q: Jakie są oczekiwania? A: W zasadzie oczekiwania są, mam takie, jakie już wcześniej opisałem, bo opisałem, co chcesz. Chcę, żeby to była aplikacja CLI. Najlepiej zróbmy ją w Pythonie, bo trochę znam Pythona i to mi będzie dla mnie najwygodniej, jeśli to będzie aplikacja w Pythonie. Aplikacja ma działać na Macu. To będzie aplikacja tylko dla mnie. Ja pracuję na Macu, więc ta aplikacja musi działać dobrze na Macu. Ale na windows też powinna. I to w zasadzie wszystko.

Q: Co może pójść nie tak?
A: jedyne co mi przychodzi na myśl że w czasie transktypcji coś się wywali i nie zapisze się materiał jeśli jest długi.
Słuchaj, i to są takie możliwe ogólne przemyślenia dotyczące tego projektu. I teraz mam prośbę do ciebie, żebyś ty przeanalizował te moje założenia i podpowiedział mi, czy są jeszcze jakieś nietypowe scenariusze, czy musimy na coś jeszcze zwrócić uwagę, nim się zabierzemy za planowanie tego projektu. Coś jeszcze ważnego widzisz, o czymś zapomniałem, czegoś ci nie napisałem. Weź to przeanalizuj i daj mi znać.
