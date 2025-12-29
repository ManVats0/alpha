❌ Pipeline Error: Expanders may not be nested inside other expanders.

StreamlitAPIException: Expanders may not be nested inside other expanders.

Traceback:
File "/mount/src/alpha/alpha.py", line 123, in <module>
    with st.expander("🔍 Model Parameters"):
         ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
📊 SYNTH_AAPL - GARCH Analysis
AttributeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 88, in exec_func_with_error_handling
    result = func()
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 579, in code_to_exec
    exec(code, module.__dict__)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
File "/mount/src/alpha/alpha.py", line 145, in <module>
    vol_pct = res['volatility'].tail(300) * 100
              ^^^^^^^^^^^^^^^^^^^^^^
