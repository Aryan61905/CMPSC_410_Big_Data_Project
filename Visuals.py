import pandas as pd
import math
import pandas as pd
import matplotlib.pyplot as plt

def to_csv_batch(src_csv, dst_dir, size=30000, index=False):
    
    # Read source csv
    df = pd.read_csv(src_csv)
    
    # Initial values
    low = 0
    high = size

    # Loop through batches
    for i in range(math.ceil(len(df) / size)):

        fname = dst_dir+'/Batch_' + str(i+1) + '.csv'
        df[low:high].to_csv(fname, index=index)
        
        # Update selection
        low = high
        if (high + size < len(df)):
            high = high + size
        else:
            high = len(df)

def main():
    to_csv_batch('Crimes_-_2001_to_Present_20231025.csv', 'Batches')

    df = pd.read_csv("Crimes_-_2001_to_Present_20231025.csv", low_memory=False)

    list(df.columns)
    newdf = df

    cross_table = pd.crosstab(newdf['Year'], newdf['Primary Type'])


    cross_table.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Crime Type Distribution Over the Years')
    plt.legend(title='Crime Type')
    #plt.show()
    plt.savefig('Graphs/Crime Type Distribution Over the Years.pdf')

    newdf1 = cross_table[['THEFT', 'CRIMINAL DAMAGE', 'NARCOTICS', 'OTHER OFFENSE', 'BATTERY', 'BURGLARY', 'ASSAULT', 'DECEPTIVE PRACTICE', 'ROBBERY', 'MOTOR VEHICLE THEFT']]

    newdf1.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title('Crime Type Distribution Over the Years (10 Most Common Types)')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.show()
    plt.savefig("Graphs/Crime Type Distribution Over the Years (10 Most Common Types).pdf")

    crime_counts = newdf.groupby(['Year', 'Primary Type']).size().unstack()
    newdf2 = crime_counts[['THEFT', 'CRIMINAL DAMAGE', 'NARCOTICS', 'OTHER OFFENSE', 'BATTERY', 'BURGLARY', 'ASSAULT', 'DECEPTIVE PRACTICE', 'ROBBERY', 'MOTOR VEHICLE THEFT']]


    newdf2.plot(kind='line', figsize=(12, 6))
    plt.title('Trends in Types of Crime Over Time (10 Most Common)')
    plt.xlabel('Year')
    plt.ylabel('Crime Count')
    plt.legend(title='Crime Type', loc='upper left', bbox_to_anchor=(1.02, 1))
    #plt.show()
    plt.savefig('Graphs/Trends in Types of Crime Over Time (10 Most Common).pdf')

    arrest_counts = newdf.groupby(['Location Description', 'Arrest']).size().unstack().fillna(0).reset_index()
    arrest_counts1 = arrest_counts.loc[arrest_counts['Location Description'].isin(['STREET', 'RESIDENCE', 'APARTMENT', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'OTHER', 'RESIDENCE-GARAGE', 'SMALL RETAIL STORE', 'RESTAURANT', 'ALLEY'])]

    arrest_counts1.plot(kind='barh', stacked=True, figsize=(12, 6), label=arrest_counts1['Location Description'])
    ax=plt.subplot(111)
    tickvalues = range(0,len(arrest_counts1['Location Description']))
    plt.yticks(ticks = tickvalues, labels = arrest_counts1['Location Description'], rotation = 'horizontal')
    plt.title('Number of Arrests by Location Description (10 Most Common)')
    plt.xlabel('Arrest Count')
    plt.ylabel('Location Description')
    plt.legend(title='Arrest', labels=['Not Arrested', 'Arrested'])
    #plt.show()
    plt.savefig('Graphs/Number of Arrests by Location Description (10 Most Common).pdf')

    newdf.groupby(['Location Description', 'Arrest']).size().sort_values(ascending=False).head(15)

    newdf["Month"] = ([x[0].split('/')[0] for x in (newdf["Date"].str.split())])
    crime_counts_by_month = newdf.groupby(['Month', 'Primary Type']).size().unstack()
    crime_counts_by_month1 = crime_counts_by_month[['THEFT', 'CRIMINAL DAMAGE', 'NARCOTICS', 'OTHER OFFENSE', 'BATTERY', 'BURGLARY', 'ASSAULT', 'DECEPTIVE PRACTICE', 'ROBBERY', 'MOTOR VEHICLE THEFT']]

    crime_counts_by_month1.plot(kind='line', figsize=(12, 6))
    plt.title('Seasonal Variations in Crime Rates by Type (10 Most Common)')
    plt.xlabel('Month')
    plt.ylabel('Crime Count')
    plt.legend(title='Crime Type', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    #plt.show()
    plt.savefig('Graphs/Seasonal Variations in Crime Rates by Type (10 Most Common).pdf')

main()

     #plt.savefig('Graphs/Crime Type Distribution Over the Years.pdf')
     #plt.savefig("Graphs/Crime Type Distribution Over the Years (10 Most Common Types).pdf")
     #plt.savefig('Graphs/Trends in Types of Crime Over Time (10 Most Common).pdf')
     #plt.savefig('Graphs/Number of Arrests by Location Description (10 Most Common).pdf')
     #plt.savefig('Graphs/Seasonal Variations in Crime Rates by Type (10 Most Common).pdf')