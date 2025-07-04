# NEW
cmake_minimum_required(VERSION 3.19)  # Newer than 3.14 for automoc & better Qt integration

project(sph)

# -------------------------------------
# Selección del compilador
# -------------------------------------
if(DEFINED ENV{CC})
    set(CMAKE_C_COMPILER $ENV{CC})
endif()

if(DEFINED ENV{CXX})
    set(CMAKE_CXX_COMPILER $ENV{CXX})
endif()

# Si aún no está definido, buscar los disponibles
if(NOT CMAKE_C_COMPILER)
    find_program(CMAKE_C_COMPILER NAMES icc icx gcc clang)
endif()

if(NOT CMAKE_CXX_COMPILER)
    find_program(CMAKE_CXX_COMPILER NAMES icpc icpx g++ clang++)
endif()

# -------------------------------------
# Estándar C++ y configuración general
# -------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE) # Ensure C++11 is met
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------
# Flags por compilador
# -------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    message(STATUS "Usando compilador Intel: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp -O3 -march=native")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    message(STATUS "Usando compilador NVIDIA HPC: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fast -mp")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Usando compilador GCC: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -ffast-math -march=native -funroll-loops -fPIC")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Usando compilador Clang: ${CMAKE_CXX_COMPILER}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -ffast-math -march=native -fPIC")

else()
    message(WARNING "Compilador desconocido: ${CMAKE_CXX_COMPILER}. Se usarán flags por defecto.")
endif()

# -------------------------------------
# Qt y OpenGL
# -------------------------------------

# NEW:
# Enable CMake's automatic handling for MOC, UIC, and RCC
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON) # Essential for finding ui_*.h and moc_*.cpp

# Find Qt5 and its components
find_package(Qt5 COMPONENTS Core Gui OpenGL Widgets REQUIRED)

# --- START DIAGNOSTIC MESSAGES ---
# message(STATUS "--- Qt5 FIND PACKAGE DIAGNOSTICS ---")
# if(Qt5_FOUND)
#     message(STATUS "Qt5 FOUND. Version: ${Qt5_VERSION_MAJOR}.${Qt5_VERSION_MINOR}.${Qt5_VERSION_PATCH}")
#     message(STATUS "Qt5 Widgets component found: ${Qt5Widgets_FOUND}")
#     message(STATUS "Qt5 Widgets library: ${Qt5Widgets_LIBRARIES}")
#     message(STATUS "Qt5 executables path: ${Qt5_BIN_DIR}")
#     message(STATUS "Qt5 include path: ${Qt5_INCLUDE_DIRS}")
#     message(STATUS "Qt5 Widgets tools path: ${Qt5Widgets_TOOL_EXECUTABLES}") # This might be empty
# else()
#     message(FATAL_ERROR "Qt5 was NOT found. Please check your Qt installation and CMake configuration.")
# endif()
# message(STATUS "--- END Qt5 FIND PACKAGE DIAGNOSTICS ---")
# --- END DIAGNOSTIC MESSAGES ---

find_package(OpenGL REQUIRED) # This will find the core OpenGL library
find_package(GLU REQUIRED)    # This will find GLU
find_package(OpenMP)

# -------------------------------------
# Archivos fuente
# -------------------------------------

# NEW: (consolidated for AUTOGEN)
set(APP_SOURCES
    src/main.cpp
    src/sph.cpp
    src/particle.cpp
    src/visualization.cpp
    src/widget.cpp
    src/sphconfig.cpp
    src/vec3.cpp
    src/sph.h            # Headers with Q_OBJECT will be processed by AUTOMOC
    src/particle.h
    src/visualization.h
    src/widget.h
    src/sphconfig.h
    src/vec3.h
    src/widget.ui        # UI forms will be processed by AUTOUIC
    # Add your .qrc files here if you have any, e.g.: src/resources.qrc
)

# Use qt_add_executable for Qt applications
add_executable(${PROJECT_NAME} ${APP_SOURCES})

target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES) # More modern way to add definitions

target_include_directories(${PROJECT_NAME} PRIVATE src ${CMAKE_BINARY_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt5::Widgets # This usually brings in Qt5::Core and Qt5::Gui automatically
    Qt5::OpenGL # Link explicitly for OpenGL capabilities provided by Qt
    ${OPENGL_LIBRARIES} # From find_package(OpenGL)
    ${GLU_LIBRARIES}    # From find_package(GLU)
    ${OpenMP_CXX_LIBRARIES}  # For OpenMP
)

