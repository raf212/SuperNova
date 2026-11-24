set(BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AdaptivePackedCellContainer/APC.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Models/GHGF.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Models/ModelIR.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/NeuromorphicTimeSpace/Fabric.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/SuperNovaComputeManager/Computer.cpp"
)

pybind11_add_module(SuperNovaBind NO_EXTRAS ${BINDING_SOURCES})
target_link_libraries(SuperNovaBind PRIVATE atomiccim_core_pic)
target_include_directories(SuperNovaBind PRIVATE headers templates)

supernova_enable_adaptivecpp(SuperNovaBind PRIVATE ${BINDING_SOURCES})
supernova_optimize_compile(SuperNovaBind)
supernova_optimize_link(SuperNovaBind)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${BINDING_SOURCES})
