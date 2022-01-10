## Set up user-specifed variable types

- Pipeline
    1. Read and save user-specified variable types.
    2. Automatically determine all variable types. If some decisions conflict with 1), go with 1).
    3. Save the final variable type decision into the dict attribute.
- All user specified variable types should be passed to the class method `store_var_type()`.
    - `store_var_type()` is only called in `fit()` and `fit_transform()`.
    -  The accepted variable types are: `'continuous'`, `'ordinal'`, `'lower_truncated'`, `'upper_truncated'`, `'twosided_truncated'`.
- `store_var_type()` parses all input variable types into the class attribute dict `var_type_dict`. `var_type_dict` may also be set up in automatic variable type specification method `set_indices()`.
    - `set_indices()` relies on the method `get_vartype_indices()`, which returns a list of variable types (one of five mentioned above). When its returned decisions conflicts with user-specified types, the latter will be used with a reminder output. 
- `var_type_dict` is used for the following tasks:
    - To be referenced and resolve conflicts with automatic decision, in `set_indices()`
    - Determine the marginal estimation method for each variable in `get_cdf_estimation_type()` method. This method is called after `set_indices()` to locate truncated variable types.
    - Determine if there is a truncated varaible, in `has_truncation()` method.
    - Output variable types in `get_vartypes()`.