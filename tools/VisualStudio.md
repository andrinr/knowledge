# Visual Studio

## Adding a library

Open project configurations, make sure *All configurations* is selected:

1. Add include directory to: C++, General, Additional Include Directories
2. Add .dll directory to: Linker, General, Additional Library Dependencies
3. Add .dll name to:  Linker, Input, Additional Dependencies

**TIP** use $(SolutionDir) variable for relative paths. 