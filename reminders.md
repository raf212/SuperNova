> 01. Before Going to Proper Disk management and vectorization :: address Fabric initialization it should be very simple seperate <br>
FabricCache -> which are initialization time based on use ae we can create a very easy starting intarface even with vectorization and disk layout `CODE:: INITCACHE_FIX_PRIO_01`

> 02. Jusrtify or remove CODE:: FAIL_SAFE_ForceNxLenMemCopy

> 03. Remove unnecessery friend class dependencies wher public facing classes can expose unnecessey privete functions to each others.

> 04. Make reuseble functions for classes where possible to remove redundency.

> 05. Current tests in `core\TestFiles` should be removed except `StaticAsserts.hpp` and those tests should be shifted with `SuperNova` Into <br>
git repo [ExternalTestSuperNova](https://github.com/raf212/ExternalTestSuperNova.git) inside a fresh branch maintainaing relation with always current beranch and add function and class specific tests.