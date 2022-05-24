"""
Module to do some date checks and transormations
"""

def date_is_anomal(date):
    """
    Checks whether a date is correct or not

    # Arguments
        date: string, date string of format day.month.year

    # Returns
        bool, True if date string is possible date, else False
    """
    day, month, year = map(int, date.split('.'))
    if day < 0 or day > 31:
        return True
    if month < 0 or month > 12:
        return True
    if year < 0 or year > 2022:
        return True

    if year == 2022 and month > 3:  # i stopped colecting at march
        return True

    return False


def change_date_format(old_format):
    """
    Turns data format from day.month.year to year-month-day

    # Arguments
        old_format: string, date string of format day.month.year

    # Returns
        new_format: string, date string of format year-month-day
    """
    day, month, year = old_format.split('.')
    new_format = f'{year}-{month}-{day}'
    return new_format
