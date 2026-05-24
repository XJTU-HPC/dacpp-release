set(LLVM_LINK_COMPONENTS support)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dacppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dpcppLib/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/parser/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rewriter/include)
include_directories(/usr/include)

set(DACPPLIB_SOURCES
  dacppLib/lib/exception.cpp
  dacppLib/include/ReconTensor.h
)
set(PARSER_SOURCES
  parser/lib/ASTParse.cpp
  parser/lib/Calc.cpp
  parser/lib/DacppStructure.cpp
  parser/lib/Param.cpp
  parser/lib/Shell.cpp
  parser/lib/Split.cpp
  parser/lib/Dacfor.cpp
)
set(REWRITER_SOURCES
  rewriter/lib/Rewriter_Buffer_new.cpp
  rewriter/lib/buffer_template_new.cpp
)

add_library(DacppLib ${DACPPLIB_SOURCES})
add_library(ParserLib ${PARSER_SOURCES})
add_library(RewriterLib ${REWRITER_SOURCES})

add_clang_executable(translator
  translator.cpp
)


set_target_properties(translator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/translator
)


clang_target_link_libraries(translator
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  DacppLib
  ParserLib
  RewriterLib
)

add_subdirectory(examples)