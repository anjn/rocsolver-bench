# p-ranav/argparse

## How to Define Arguments

In `p-ranav/argparse`, you define command-line arguments using the `add_argument()` method of the `ArgumentParser` object. Method chaining allows for concise argument configuration.

### Basic `add_argument()` Method

`add_argument()` takes one or more argument names (e.g., `-v`, `--verbose`). Multiple names can be passed as a variadic list. It supports positional, optional, and compound arguments.

```
program.add_argument("foo");          // Positional argument
program.add_argument("-v", "--verbose"); // Optional argument (parameter packing)
```

### Positional Arguments

Positional arguments are defined by passing their name directly to `add_argument()`. These must be specified in a particular order on the command line.

```
program.add_argument("square")
.help("display the square of a given integer")
.scan<'i', int>();
```

`.help()` displays a description of the argument in the help message. `.scan<'i', int>()` converts the string argument from the command line to a specified type (e.g., `int`). This explicit type scanning is essential for correctly parsing numeric types. Omitting it can lead to runtime errors.

### Optional Arguments

Optional arguments are defined with names starting with `-` or `--`, and their order on the command line is usually not important.

```
program.add_argument("--verbose")
.help("increase output verbosity")
.default_value(false)
.implicit_value(true);
```

`.default_value(false)` sets a default value if the argument is not provided. `.implicit_value(true)` treats the argument as a flag, setting its value to `true` if present without an explicit value. `.flag()` is a shorthand for `default_value(false).implicit_value(true)`.

You can make an optional argument required using `.required()`. If not provided, `parse_args()` will throw an exception. Combining `default_value()` and `required()` is also possible.

`.append()` collects all values into a `std::vector` if the same optional argument is specified multiple times. An `action` lambda combined with `.append()` and `.nargs(0)` can be used to increment a value each time the argument is repeated.

`.nargs(N)` associates `N` command-line arguments with a single action, collecting them into a list. Variable-length lists are also supported using `nargs(min, max)` or predefined patterns like `argparse::nargs_pattern::any`, `at_least_one`, and `optional`.

`program.add_mutually_exclusive_group(required = false)` ensures that only one argument within the group is specified. Setting `required = true` enforces that at least one argument from the group must be provided.

### Overview of Argument Definition Methods and Options

The main methods that can be chained to `add_argument()` control the argument's behavior. Method chaining allows for concise configuration and improves code readability.

|Method Name|Function|Parameters|Return Type|Example Usage|
|---|---|---|---|---|
|`.help()`|Sets the argument's description, displayed in help messages.|`std::string` (description text)|`ArgumentParser&`|`.help("Description text")`|
|`.scan<'c', Type>()`|Defines the logic to convert command-line strings to a specific type.|`'c'` (format specifier, e.g., `'i'` for int), `Type` (target type)|`ArgumentParser&`|`.scan<'i', int>()`|
|`.default_value()`|Sets a default value if the argument is not specified.|`Type` (default value)|`ArgumentParser&`|`.default_value(false)`|
|`.implicit_value()`|Defines the value set if the argument is present without a value (for flags).|`Type` (implicit value)|`ArgumentParser&`|`.implicit_value(true)`|
|`.flag()`|Shorthand for `default_value(false).implicit_value(true)` to define a boolean flag.|None|`ArgumentParser&`|`.flag()`|
|`.required()`|Makes an optional argument mandatory.|None|`ArgumentParser&`|`.required()`|
|`.append()`|Collects all values into a `std::vector` if the same argument is specified multiple times.|None|`ArgumentParser&`|`.append()`|
|`.nargs()`|Specifies the number of command-line arguments associated with an argument.|`int N` (fixed number), `int min, int max` (range), `argparse::nargs_pattern` (pattern)|`ArgumentParser&`|`.nargs(3)`, `.nargs(1, 5)`, `.nargs(argparse::nargs_pattern::any)`|
|`add_mutually_exclusive_group()`|Creates a group where only one argument can be specified from multiple options.|`bool required` (whether at least one from the group is required)|`ArgumentParser&`|`program.add_mutually_exclusive_group(true)`|

## How to Retrieve Argument Values

### Parsing Arguments (`parse_args()`)

After defining arguments, you must call the `parse_args()` method of the `ArgumentParser` object to actually parse the command-line arguments (typically `argc` and `argv`).

If required arguments are missing, invalid values are specified, or other parsing errors occur, `parse_args()` will throw a `std::exception`. It is highly recommended to wrap this call in a `try-catch` block to handle errors gracefully and display the automatically generated help message using `std::cerr << program;`.

```
int main(int argc, char *argv) {
    argparse::ArgumentParser program("program_name");
    //... Argument definitions...
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl; // Display error message
        std::cerr << program; // Display help message
        return 1; // Exit with error
    }
    //... Retrieve and use parsed values...
}
```

### Retrieving Values with Type Specification (`.get<Type>()`)

The most common method for retrieving values is `get<Type>("argument_name")`. This method returns the value of the specified type (`Type`).

```
auto input = program.get<int>("square"); // (If.scan<'i', int> was used previously)
auto color = program.get<std::string>("--color");
```

For numeric types, you must register the type conversion logic using `.scan<'i', int>()` (or appropriate type specification) when defining the argument with `add_argument()`. Without this, `get<Type>()` may not work correctly or could lead to unexpected runtime errors.

### Checking Flag Values

The value of flag arguments (defined with `.flag()` or similar) can be accessed directly using the `` operator. This operator typically returns a `bool` value, `true` if the flag is present.

```
if (program["--verbose"] == true) {
    std::cout << "Verbosity enabled" << std::endl;
}
```

### Checking Argument Presence (`.present()`, `.is_used()`)

`p-ranav/argparse` provides methods to check the presence of arguments and whether they were explicitly provided by the user.

- **`.present("argument_name")`**: This method checks if a specific optional argument was provided on the command line. The return type is `std::optional<T>`, which contains the value if the argument is present, or an empty `std::optional` otherwise. This is particularly useful for safely accessing optional arguments that do not have default values.
    
    ```
    if (auto fn = program.present("-o")) {
        // Use *fn to access the value (retrieve value from std::optional<T>)
    }
    ```
    
- **`.is_used("argument_name")`**: This method returns a `bool` indicating whether the argument was explicitly provided by the user on the command line. This is useful even if the argument has a default value, to distinguish if the user overrode it.
    
    ```
    auto explicit_color = program.is_used("--color"); // true if user provided --color, false if default value was used
    ```
    

These methods differ from `get<Type>()` and the `` operator, providing more detailed information about argument provision and user intent.

### Comparison of Argument Value Retrieval Methods

There are multiple ways to retrieve argument values, each best suited for different scenarios.

| Method Name                 | Purpose                                                                                             | Return Type           | Suitable Usage Scenario                                                                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `program.get<Type>("name")` | Retrieves the parsed argument value in the specified type.                                          | Specified `Type`      | When the argument is always present and type conversion is guaranteed (e.g., positional arguments, `required()` optional arguments, numeric arguments with `.scan()` applied). |
| `program["--name"]`         | Checks the state of a boolean flag argument.                                                        | `bool`                | For simple checks of flag presence defined with `.flag()` or `default_value(false).implicit_value(true)`.                                                                      |
| `program.present("name")`   | Checks if an optional argument was provided on the command line and retrieves its value if present. | `std::optional<Type>` | To safely check for the presence of optional arguments without default values, and access their values only if they exist.                                                     |
| `program.is_used("name")`   | Checks if the user explicitly provided an argument, even if it has a default value.                 | `bool`                | When an argument has a default value, but you need to differentiate if the user overrode that value.                                                                           |

## Practical Example and Best Practices

The following C++ code example combines the key argument definition and value retrieval features discussed in previous sections. This example specifically demonstrates how to use multiple argument types, including positional arguments, optional arguments (flags, value-taking), required options, options with default values, and list arguments.

```
#include <argparse/argparse.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> // Required for std::exception

int main(int argc, char *argv) {
    argparse::ArgumentParser program("my_app", "1.0.0"); // Program name and version

    // 1. Positional argument: Required input file
    program.add_argument("input_file")
  .help("Path to the input file.")
  .scan<std::string>(); // Not strictly required for string, but explicit

    // 2. Optional argument: Output file (required)
    program.add_argument("-o", "--output")
  .required() // This option is mandatory
  .help("Specify the output file path.");

    // 3. Optional argument: Verbose mode (flag)
    program.add_argument("-v", "--verbose")
  .flag() // Shorthand for default_value(false).implicit_value(true)
  .help("Enable verbose output.");

    // 4. Optional argument: Numeric type (port number)
    program.add_argument("--port")
  .help("Specify the port number.")
  .default_value(8080) // Default value
  .scan<'i', int>(); // Explicit conversion to numeric type

    // 5. Optional argument: List (tags)
    program.add_argument("--tags")
  .help("Comma-separated list of tags.")
  .nargs(argparse::nargs_pattern::at_least_one) // At least one value required
  .append(); // Collects values into a list if specified multiple times

    // 6. Mutually exclusive group: Mode selection
    auto group = program.add_mutually_exclusive_group(true); // One from the group is required
    group.add_argument("--mode-read")
  .flag()
  .help("Set application to read mode.");
    group.add_argument("--mode-write")
  .flag()
  .help("Set application to write mode.");

    // Argument parsing and error handling
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl; // Display error message
        std::cerr << program; // Display auto-generated help message
        return 1; // Exit with error
    }

    // Retrieve and use parsed values
    std::string input_file = program.get<std::string>("input_file");
    std::string output_file = program.get<std::string>("--output");
    bool verbose_mode = program["--verbose"];
    int port_number = program.get<int>("--port");

    std::cout << "Input File: " << input_file << std::endl;
    std::cout << "Output File: " << output_file << std::endl;
    std::cout << "Verbose Mode: " << (verbose_mode? "Enabled" : "Disabled") << std::endl;
    std::cout << "Port Number: " << port_number << std::endl;

    // Check presence and use of optional arguments
    if (program.present("--tags")) { // Check if tags were specified
        std::vector<std::string> tags = program.get<std::vector<std::string>>("--tags");
        std::cout << "Tags: ";
        for (const auto& tag : tags) {
            std::cout << tag << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "No tags specified." << std::endl;
    }

    // Check if default value was used or explicitly provided by user
    if (program.is_used("--port")) {
        std::cout << "Port number was explicitly provided by user." << std::endl;
    } else {
        std::cout << "Default port number used." << std::endl;
    }

    // Check mutually exclusive modes
    if (program["--mode-read"]) {
        std::cout << "Application is in READ mode." << std::endl;
    } else if (program["--mode-write"]) {
        std::cout << "Application is in WRITE mode." << std::endl;
    }

    return 0;
}
```

### Tips for Robust Argument Handling

Tips for effectively using `p-ranav/argparse`:

- **Exception Handling:** `parse_args()` throws exceptions, so wrap it in `try-catch` to provide error messages and help display.
    
- **Help Messages:** Use `.help()` to clearly describe the purpose and usage examples of arguments, aiding user understanding.
    
- **Type Conversion (`.scan<>`):** Always define explicit type conversion with `.scan<>()` for numeric and other non-string arguments to prevent runtime errors.
    
- **Default Values:** Set default values where possible to reduce user input burden and increase program flexibility.
    
- **`present()` and `is_used()`:** Use these methods appropriately when you need to distinguish between the presence of an optional argument and whether it was explicitly provided by the user.
    
- **Mutually Exclusive Groups:** Utilize `add_mutually_exclusive_group()` to enforce logical constraints where multiple options should not be specified simultaneously.