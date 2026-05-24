cmake_minimum_required(VERSION 3.8)
project(activeslam_resource)

find_package(ament_cmake REQUIRED)

install(DIRECTORY maps models
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()
