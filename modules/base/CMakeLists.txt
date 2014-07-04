message(${CMAKE_CURRENT_SOURCE_DIR})

# ----------------------------------------------------------------------------
set(BACKPROPAGATIONNETWORK_LIB_NAME bpn)
SET(BACKPROPAGATIONNETWORK_VERSION_MAJOR "1")
SET(BACKPROPAGATIONNETWORK_VERSION_MINOR "0")
SET(BACKPROPAGATIONNETWORK_VERSION_PATCH "0")

set(BACKPROPAGATIONNETWORK_VERSION_STRING 
	"${BACKPROPAGATIONNETWORK_VERSION_MAJOR}."
	"${BACKPROPAGATIONNETWORK_VERSION_MINOR}."
	"${BACKPROPAGATIONNETWORK_VERSION_PATCH}" )
SET(BACKPROPAGATIONNETWORK_LIB_SOVERSION ${BACKPROPAGATIONNETWORK_VERSION_MAJOR}) 
# ----------------------------------------------------------------------------
message("Building BackPropagationNetWork library (shared and static)...")
add_library(${BACKPROPAGATIONNETWORK_LIB_NAME} SHARED STATIC
	BackPropagationNetwork.hpp
	BackPropagationNetwork.cpp)

# ----------------------------------------------------------------------------  
message("Installing BackPropagationNetWork library ...")
install(
	TARGETS 	${BACKPROPAGATIONNETWORK_LIB_NAME}
	INCLUDES    DESTINATION ${INCLUDE_HEADER_DIRECTORY}		
	RUNTIME     DESTINATION ${RUNTIME_OUTPUT_DIRECTORY}
	LIBRARY     DESTINATION ${LIBRARY_OUTPUT_DIRECTORY}
	ARCHIVE     DESTINATION ${LIBRARY_OUTPUT_DIRECTORY}
	# COMPONENT 	library	
)
# ----------------------------------------------------------------------------  
# message("Installing BackPropagationNetWork header files ...") 
# install(
	# FILES 
		# dnn.hpp 
		# BackPropagationNetWork.hpp 
    # DESTINATION ${INCLUDE_HEADER_DIRECTORY}
    # # DESTINATION ${CMAKE_HEADER_INCLUDE_DIRECTORY}
# )
# ----------------------------------------------------------------------------
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("CUDA is found on this system.")
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_NVCC_FLAGS "
		-gencode;arch=compute_35,code=sm_35; 
		-gencode;arch=compute_30,code=sm_30; 
		-gencode;arch=compute_20,code=sm_20; 
		# -gencode;arch=compute_11,code=sm_11; 
		# -gencode;arch=compute_12,code=sm_12;
		# -gencode;arch=compute_13,code=sm_13;
		")

	# add -Wextra compiler flag for gcc compilations
	if (UNIX)
		# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--disable-warnings;")
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	endif (UNIX)
else(CUDA_FOUND)
    message(FATAL_ERROR "CUDA is not installed on this system.")
endif()
# ----------------------------------------------------------------------------
include_directories(${INCLUDE_HEADER_DIRECTORY})
cuda_add_executable(bpn_empty bpn_empty.cpp)
target_link_libraries(bpn_empty 
	${BACKPROPAGATIONNETWORK_LIB_NAME})

add_test(TEST_BPN_EMPTY
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bpn_empty
	)