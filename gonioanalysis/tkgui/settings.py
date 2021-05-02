import gonioanalysis.settings as settings

DEFAULT_SAVEFN = 'gonioanalysis-tkgui-settings.json'

def set(*args, fn=DEFAULT_SAVEFN, **kwargs):
    return settings.set(*args, fn=fn, **kwargs)

def get(*args, fn=DEFAULT_SAVEFN, **kwargs):
    return settings.get(*args, fn=fn, **kwargs)

