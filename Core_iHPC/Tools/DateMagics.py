"""
.. module:: DateMagics
   :platform: Unix
   :synopsis: Everything needed to convert dates.

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
                  

"""
import pytz
import sys
import datetime as dt
###########################################################################################
def DateMagics(date, timezone):
    """
    This function inputs a naive datetime and a timezone, and returns 3 'UTC' 'AEST' 'AEDT' datetime.
    timezones should be 'UTC' 'AEST' 'AEDT'
     
    Parameters
    ---------- 
        date : datetime
            naive datetime .
        timezone : str
            timezone of the input naive datetime.
    Returns
    -------
        dateUTC : datetime
            date in UTC timezone.        
        dateAEST : datetime
            date in AEST timezone.        
        dateAEDT : datetime
            date in AEDT timezone.        
    """
    aedt = pytz.timezone('Australia/Sydney')
    aest = pytz.timezone('Australia/Brisbane')
    
    if timezone not in ['UTC', 'AEST', 'AEDT']:
        sys.exit("tz not implemented")
    
    if date.tzinfo is None:
        if timezone == 'UTC':
            datelocal = pytz.utc.localize(date)
        if timezone == 'AEST':
            datelocal = aest.localize(date)
        if timezone == 'AEDT':
            datelocal = aedt.localize(date)
    else:
        datelocal = date
    
    dateUTC  = datelocal.astimezone(pytz.utc)
    dateAEST = datelocal.astimezone(aest)
    dateAEDT = datelocal.astimezone(aedt)
    
    return dateUTC, dateAEST, dateAEDT 
###########################################################################################
def DateMagicsList(date, timezone):
    """
    This function inputs a list of naive datetime and a timezone, and returns 3 list of 'UTC' 'AEST' 'AEDT' datetime.
    timezones should be 'UTC' 'AEST' 'AEDT'
     
    Parameters
    ---------- 
        date : datetime
            list of naive datetime .
        timezone : str
            timezone of the input naive datetime.
    Returns
    -------
        dateUTC : datetime
            list of dates in UTC timezone.        
        dateAEST : datetime
            list of dates in AEST timezone.        
        dateAEDT : datetime
            list of dates in AEDT timezone.        
    """
    aedt = pytz.timezone('Australia/Sydney')
    aest = pytz.timezone('Australia/Brisbane')
    
    if timezone not in ['UTC', 'AEST', 'AEDT']:
        sys.exit("tz not implemented")
    
    if timezone == 'UTC':
        datelocal = [pytz.utc.localize(Date) for i,Date in enumerate(date)]
    if timezone == 'AEST':
        datelocal = [aest.localize(Date) for i,Date in enumerate(date)]
    if timezone == 'AEDT':
        datelocal = [aedt.localize(Date) for i,Date in enumerate(date)]
        
    dateUTC  = [date.astimezone(pytz.utc) for i,date in enumerate(datelocal)]
    dateAEST = [date.astimezone(aest) for i,date in enumerate(datelocal)]
    dateAEDT = [date.astimezone(aedt) for i,date in enumerate(datelocal)]
    
    return dateUTC, dateAEST, dateAEDT 
###########################################################################################
def DateMagicsConvert(date, timezone, inputformat):
    """
    This function inputs a list of string datetime, a timezone and the datestring format, and returns 3 list of 'UTC' 'AEST' 'AEDT' datetime.
    timezones should be 'UTC' 'AEST' 'AEDT'
    input string format has to follow the python datetime format conversion 
     
    Parameters
    ---------- 
        date : datetime
            naive datetime .
        timezone : str
            timezone of the input naive datetime.
        inputformat: str
            string format of the date string input      
    Returns
    -------
        dateUTC : datetime
            list of dates in UTC timezone.        
        dateAEST : datetime
            list of dates in AEST timezone.        
        dateAEDT : datetime
            list of dates in AEDT timezone.        
    """
    aedt = pytz.timezone('Australia/Sydney')
    aest = pytz.timezone('Australia/Brisbane')
    
    if timezone not in ['UTC', 'AEST', 'AEDT']:
        sys.exit("tz not implemented")
    
    if timezone == 'UTC':
        datelocal = pytz.utc.localize(dt.datetime.strptime(date, inputformat))
    if timezone == 'AEST':
        datelocal = aest.localize(dt.datetime.strptime(date, inputformat))
    if timezone == 'AEDT':
        datelocal = aedt.localize(dt.datetime.strptime(date, inputformat))
        
    dateUTC  = datelocal.astimezone(pytz.utc) 
    dateAEST = datelocal.astimezone(aest) 
    dateAEDT = datelocal.astimezone(aedt) 
    
    return dateUTC, dateAEST, dateAEDT 
###########################################################################################
def DateMagicsListConvert(date, timezone, inputformat):
    """
    This function inputs a list of string datetime, a timezone and the datestring format, and returns 3 list of 'UTC' 'AEST' 'AEDT' datetime.
    timezones should be 'UTC' 'AEST' 'AEDT'
    input string format has to follow the python datetime format conversion 
     
    Parameters
    ---------- 
        date : datetime
            list of naive datetime .
        timezone : str
            timezone of the input naive datetime.
        inputformat: str
            string format of the date string input      
    Returns
    -------
        dateUTC : datetime
            list of dates in UTC timezone.        
        dateAEST : datetime
            list of dates in AEST timezone.        
        dateAEDT : datetime
            list of dates in AEDT timezone.        
    """
    aedt = pytz.timezone('Australia/Sydney')
    aest = pytz.timezone('Australia/Brisbane')
    
    if timezone not in ['UTC', 'AEST', 'AEDT']:
        sys.exit("tz not implemented")
    
    if timezone == 'UTC':
        datelocal = [pytz.utc.localize(dt.datetime.strptime(Date, inputformat)) for i,Date in enumerate(date)]
    if timezone == 'AEST':
        datelocal = [aest.localize(dt.datetime.strptime(Date, inputformat)) for i,Date in enumerate(date)]
    if timezone == 'AEDT':
        datelocal = [aedt.localize(dt.datetime.strptime(Date, inputformat)) for i,Date in enumerate(date)]
        
    dateUTC  = [date.astimezone(pytz.utc) for i,date in enumerate(datelocal)]
    dateAEST = [date.astimezone(aest) for i,date in enumerate(datelocal)]
    dateAEDT = [date.astimezone(aedt) for i,date in enumerate(datelocal)]
    
    return dateUTC, dateAEST, dateAEDT 
###########################################################################################

if __name__ == '__main__':


    import datetime as dt
    
    date = dt.datetime.now()
    dateUTC, dateAEST, dateAEDT = DateMagics(date, 'AEDT')
    print (dateUTC, dateAEST, dateAEDT)
    print (dateUTC - dateAEST,  dateUTC - dateAEDT)
    dateUTC, dateAEST, dateAEDT = DateMagics(dateAEST, 'AEDT')
    print (dateUTC, dateAEST, dateAEDT)
    print (dateUTC - dateAEST,  dateUTC - dateAEDT)
    
