import streamlit.report_thread as ReportThread
from streamlit.server.server import Server
import time

class SessionState():
    """SessionState: Add per-session state to Streamlit."""
    # def __init__(self, **kwargs):
    #     """A new SessionState object.
    #     Parameters
    #     ----------
    #     **kwargs : any
    #         Default values for the session state.
    #     Example
    #     -------
    #     >>> session_state = SessionState(user_name='', favorite_color='black')
    #     >>> session_state.user_name = 'Mary'
    #     ''
    #     >>> session_state.favorite_color
    #     'black'
    #     """
    #     for key, val in kwargs.items():
    #         setattr(self, key, val)

    def __init__(self, session_objects_dict):
        self.session_objects_dict = session_objects_dict
        # for key, val in session_objects_dict.items():
        #     setattr(self, key, val)


def get(**kwargs):
    """Gets a SessionState object for the current session.
    Creates a new object if necessary.
    Parameters
    ----------
    **kwargs : any
        Default values you want to add to the session state, if we're creating a
        new one.
    Example
    -------
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    ''
    >>> session_state.user_name = 'Mary'
    >>> session_state.favorite_color
    'black'
    Since you set user_name above, next time your script runs this will be the
    result:
    >>> session_state = get(user_name='', favorite_color='black')
    >>> session_state.user_name
    'Mary'
    """
    # Hack to get the session object from Streamlit.
    time.sleep(0.1)
    session_id = ReportThread.get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError('Could not get Streamlit session object.')

    this_session = session_info.session

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(**kwargs)

    return this_session._custom_session_state


def get1(session_objects_dict):
    # Hack to get the session object from Streamlit.
    time.sleep(0.4)
    session_id = ReportThread.get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError('Could not get Streamlit session object.')

    this_session = session_info.session

    # Got the session object! Now let's attach some state into it.

    if not hasattr(this_session, '_custom_session_state'):
        this_session._custom_session_state = SessionState(session_objects_dict)

    return this_session._custom_session_state