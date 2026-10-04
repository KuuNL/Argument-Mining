# Pilot errors: pilot-k4

```json
{
 "run": "pilot-k4",
 "error_kinds": {
  "invented Inference": 167,
  "right Inference": 99,
  "missed Inference": 73,
  "right Rephrase": 61,
  "Rephrase read as Inference": 47,
  "missed Rephrase": 46,
  "invented Rephrase": 37,
  "Inference, wrong direction": 24,
  "invented Conflict": 18,
  "Inference read as Rephrase": 17,
  "missed Conflict": 14,
  "right Conflict": 7,
  "Conflict read as Rephrase": 4,
  "Inference read as Conflict": 2,
  "Rephrase read as Conflict": 2
 },
 "median_turn_distance": {
  "missed": 1,
  "invented": 2.0,
  "right": 1
 }
}
```

## nodeset20299 (11 propositions)
```text
T1  Audience Member: Why has it taken until 15 February for the government to impose quarantine at the border
    [P1] it has taken until 15 February for the government to impose quarantine at the border because xxx
T2  Audience Member: Why haven't lessons been learned from Australia who closed their border in the first wave and now have low levels of infection
    [P2] lessons haven't been learned from Australia who closed their border in the first wave and now have low levels of infection because xxx
T3  Ian Murray: It is a great question
    [P3] it is a great question
T4  Ian Murray: a question we have been asking in Parliament for the last year
    [P4] a question we have been asking in Parliament for the last year
T5  Ian Murray: We're an island nation and most other island nations across the world did better than us
    [P5] we're an island nation and most other island nations across the world did better than us
T6  Ian Murray: they took the decision early to stop incoming traffic
    [P6] most other island nations took the decision early to stop incoming traffic
T7  Ian Murray: We had the South Africa strain for 50-odd days
    [P7] we had the South Africa strain for 50-odd days
T8  Ian Murray: It is only now we take the decision to close down the borders
    [P8] it is only now we take the decision to close down the borders
T9  Ian Murray: It's too late
    [P9] closing borders now is too late
T10  Ian Murray: There was a study by Aberdeen University which showed most of the deaths happened because of the inward variants coming in to Scotland and the rest of the United Kingdom
    [P10] there was a study by Aberdeen University which showed most of the deaths happened because of the inward variants coming in to Scotland and the rest of the United Kingdom
T11  Ian Murray: It seems, again, where both governments have been too slow to react
    [P11] it seems, again, where both governments have been too slow to react
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P5 -> P1 Rephrase | invented Rephrase |
| - | P8 -> P1 Rephrase | invented Rephrase |
| - | P4 -> P3 Inference | invented Inference |
| P6 -> P5 Inference | P6 -> P5 Inference | right Inference |
| P7 -> P9 Inference | P7 -> P9 Inference | right Inference |
| P8 -> P9 Inference | P8 -> P9 Inference | right Inference |
| P10 -> P9 Inference | P10 -> P9 Inference | right Inference |
| - | P11 -> P9 Rephrase | invented Rephrase |
| P10 -> P11 Inference | - | missed Inference |

## nodeset20306 (13 propositions)
```text
T1  Audience Member: after Brexit which divided us, now with this pandemic, do we really want independence to resow division and cripple our economy
    [P1] after Brexit which divided us, now with this pandemic, do or don't we really want independence to resow division and cripple our economy
T2  Audience Member: It was once in a lifetime we were told, and we voted
    [P2] it was once in a lifetime we were told, and we voted
T3  Michael Forsyth: I took the view that if the SNP had been able to get the support of the Scottish electorate then there should be a referendum
    [P3] Michael Forsyth took the view that if the SNP had been able to get the support of the Scottish electorate then there should be a referendum
T4  Michael Forsyth: I said that many years ago
    [P4] Michael Forsyth said that if the SNP had been able to get the support of the Scottish electorate then there should be a referendum many years ago
T5  Michael Forsyth: we had a referendum
    [P5] we had a referendum
T6  Michael Forsyth: it was once in a generation referendum
    [P6] it was once in a generation referendum
T7  Michael Forsyth: The things about people who say they believe in democracy, they don't accept the referendum
    [P7] the things about people who say they believe in democracy, they don't accept the referendum
T8  Michael Forsyth: the Scottish Parliament, the elections to the Scottish Parliament, the duties of the Scottish Parliament are about health, education, employment, the things that really matter to people
    [P8] the Scottish Parliament, the elections to the Scottish Parliament, the duties of the Scottish Parliament are about health, education, employment, the things that really matter to people
T9  Michael Forsyth: They're not about constitutional matters
    [P9] the duties of the Scottish Parliament are not about constitutional matters
T10  Michael Forsyth: They're reserve to the United Kingdom Parliament
    [P10] they're reserve to the United Kingdom Parliament
T11  Michael Forsyth: The reason the SNP are campaigning on this constitutional issue is because they're not interested in devolution, they're interested in the destruction of the United Kingdom
    [P11] the reason the SNP are campaigning on this constitutional issue is because they're not interested in devolution, they're interested in the destruction of the United Kingdom
T12  Michael Forsyth: they don't want the election to be about their record on health or any other areas because it is abysmal
    [P12] they don't want the election to be about their record on health or any other areas because it is abysmal
T13  Michael Forsyth: When I left office, Scottish education was way ahead of England, now it is way behind
    [P13] when Michael Forsyth left office, Scottish education was way ahead of England, now it is way behind
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P2 -> P1 Inference | invented Inference |
| P3 -> P1 Rephrase | - | missed Rephrase |
| P6 -> P2 Rephrase | P6 -> P2 Rephrase | right Rephrase |
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| - | P5 -> P7 Inference | invented Inference |
| - | P6 -> P7 Inference | invented Inference |
| P9 -> P8 Inference | P9 -> P8 Rephrase | Inference read as Rephrase |
| - | P10 -> P9 Inference | invented Inference |
| - | P9 -> P11 Inference | invented Inference |
| P12 -> P11 Inference | P12 -> P11 Inference | right Inference |
| P13 -> P12 Inference | P13 -> P12 Inference | right Inference |

## nodeset20309 (12 propositions)
```text
T1  Michael Forsyth: in the middle of a pandemic, when we have all the consequences in terms of lost jobs and the rest
    [P1] in the middle of a pandemic, when we have all the consequences in terms of lost jobs and the rest
T2  Michael Forsyth: people should be thinking that we should divide our country
    [P2] people should be thinking that we should divide our country
T3  Michael Forsyth: putting together the damage which has been done by this virus
    [P3] people should be putting together the damage which has been done by this virus
T4  Michael Forsyth: getting our economy growing again in order to pay down the debt and to pay jobs
    [P4] people should be thinking about getting our economy growing again in order to pay down the debt and to pay jobs
T5  Michael Forsyth: Scotland is not the SNP
    [P5] Scotland is not the SNP
T6  Michael Forsyth: The Scottish Nationalist Party shouldn't take the electorate for granted and assume they are going to get a majority
    [P6] the Scottish Nationalist Party shouldn't take the electorate for granted and assume they are going to get a majority
T7  Michael Forsyth: They should concentrate on health and education
    [P7] the SNP should concentrate on health and education
T8  Jeane Freeman: We are not insisting or taking for granted we would get a majority at all
    [P8] the SNP is not insisting or taking for granted we would get a majority at all
T9  Jeane Freeman: the people in Scotland who are talking -- political parties who are talking most about independence are your own and Labour
    [P9] the people in Scotland who are talking -- political parties who are talking most about independence are your own and Labour
T10  Jeane Freeman: What we're talking about is a pandemic and how you rebuild Scotland
    [P10] what the SNP is talking about is a pandemic and how you rebuild Scotland
T11  Jeane Freeman: there is a majority in the next Parliamentary election on 6 May, 21 polls in a row, people saying that they want to be independent
    [P11] there is a majority in the next Parliamentary election on 6 May, 21 polls in a row, people saying that they want to be independent
T12  Jeane Freeman: people in Scotland have the right to choose whether or not they want to have a referendum
    [P12] people in Scotland have the right to choose whether or not they want to have a referendum
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P1 -> P2 Inference | invented Inference |
| P1 -> P3 Inference | - | missed Inference |
| P3 -> P2 Conflict | P3 -> P2 Rephrase | Conflict read as Rephrase |
| - | P9 -> P2 Conflict | invented Conflict |
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| P5 -> P6 Inference | - | missed Inference |
| - | P7 -> P6 Rephrase | invented Rephrase |
| P8 -> P6 Conflict | P8 -> P6 Conflict | right Conflict |
| P10 -> P7 Conflict | - | missed Conflict |
| - | P10 -> P8 Rephrase | invented Rephrase |
| P12 -> P11 Rephrase | P11 -> P12 Inference | Rephrase read as Inference |

## nodeset20311 (16 propositions)
```text
T1  Stephen Fitzpatrick: I cannot understand for the life of me why 12 months ago this was not a policy we instituted straightaway
    [P1] Stephen Fitzpatrick cannot understand for the life of him why 12 months ago this was not a policy we instituted straightaway
T2  Stephen Fitzpatrick: When I hear people talk about the policy today, what I hear is they don't understand why we didn't do this a year ago
    [P2] When Stephen Fitzpatrick hears people talk about the policy today, what he hears is they don't understand why we didn't institute the policy straightaway a year ago
T3  Stephen Fitzpatrick: But our politicians are not paid to peddle paranoia or feed on our fears
    [P3] Our politicians are not paid to peddle paranoia or feed on our fears
T4  Stephen Fitzpatrick: they are paid to lead
    [P4] Our politicians are paid to lead
T5  Stephen Fitzpatrick: When I look at all of the news today, I see signs of hope everywhere
    [P5] When Stephen Fitzpatrick looks at all of the news today, he sees signs of hope everywhere
T6  Stephen Fitzpatrick: We have better therapeutics, we have falling cases, we have a better understanding of infection rates, we have 90 percent of the most vulnerable people in the country vaccinated
    [P6] We have better therapeutics, we have falling cases, we have a better understanding of infection rates, we have 90 percent of the most vulnerable people in the country vaccinated
T7  Stephen Fitzpatrick: I ask what are people afraid of
T8  Fiona Bruce: It is not quite that yet
    [P7] It is not quite that yet with the vaccination
T9  Stephen Fitzpatrick: 90 percent of the over 70s have been vaccinated. 50 percent of the panellists in this room have been vaccinated
    [P8] 90 percent of the over 70s have been vaccinated. 50 percent of the panellists in this room have been vaccinated
T10  Stephen Fitzpatrick: So when I think about what it is that politicians are trying to encourage
    [P9] What it is that politicians are trying to encourage
T11  Stephen Fitzpatrick: I say we weren't ready nor Covid-19, we weren't ready for lock down and we have to be ready for life after lock down
    [P10] We weren't ready for Covid-19, we weren't ready for lock down and we have to be ready for life after lock down
T12  Stephen Fitzpatrick: When I hear politicians talking about banning all inbound travel
    [P11] Stephen Fitzpatrick hears politicians talking about banning all inbound travel
T13  Fiona Bruce: That's the question
    [P12] The question is about banning all inbound travel
T14  Fiona Bruce: Do you think that should have happen now?
    [P13] Does or doesn't Stephen Fitzpatrick think that banning inbound travel should have happened now?
T15  Stephen Fitzpatrick: I understand if a political leader says to me we have taken a wisk-based approach
    [P14] Stephen Fitzpatrick understands if a political leader says to him we have taken a wisk-based approach
T16  Stephen Fitzpatrick: these are 33 risky countries and we think this risk is worth protecting against
    [P15] These are 33 risky countries and we think this risk is worth protecting against
T17  Stephen Fitzpatrick: I think be it's a popular decision to ban all travel.
    [P16] It's a popular decision to ban all travel
```
| Gold | Predicted | Verdict |
|---|---|---|
| P1 -> P2 Inference | P2 -> P1 Rephrase | Inference read as Rephrase |
| P3 -> P4 Inference | P4 -> P3 Inference | Inference, wrong direction |
| P5 -> P6 Inference | P6 -> P5 Inference | Inference, wrong direction |
| P7 -> P6 Conflict | P7 -> P6 Conflict | right Conflict |
| P6 -> P8 Inference | P8 -> P6 Inference | Inference, wrong direction |
| P8 -> P7 Conflict | P8 -> P7 Conflict | right Conflict |
| - | P9 -> P10 Inference | invented Inference |
| P12 -> P11 Inference | P12 -> P11 Rephrase | Inference read as Rephrase |
| P12 -> P13 Inference | - | missed Inference |
| P16 -> P13 Inference | P16 -> P13 Rephrase | Inference read as Rephrase |
| P14 -> P15 Inference | P15 -> P14 Inference | Inference, wrong direction |

## nodeset20329 (9 propositions)
```text
T1  Angela Haggerty: The SNP as a single issue party can provide a home
    [P1] the SNP as a single issue party can provide a home
T2  Angela Haggerty: the SNP is in government in Scotland as well
    [P2] the SNP is in government in Scotland as well
T3  Angela Haggerty: it then has to come up with policies and it has to take positions on things
    [P3] the SNP then has to come up with policies and it has to take positions on things
T4  Angela Haggerty: it is a tightrope to walk and the SNP can't continue to do that long term
    [P4] it is a tightrope to walk and the SNP can't continue to do that long term
T5  Angela Haggerty: It was inevitable something like this would happen
    [P5] it was inevitable something like this would happen
T6  Angela Haggerty: the SNP was counting on it happening after independence not before it
    [P6] the SNP was counting on it happening after independence not before it
T7  Stephen Fitzpatrick: What I thought I heard is because a politician is particularly popular at the moment, then it may be okay if they breach a ministerial code, they would not resign where otherwise they normally would
    [P7] what Stephen Fitzpatrick thought he heard is that because a politician is particularly popular at the moment, then it may be okay if they breach a ministerial code, they would not resign where otherwise they normally would
T8  Stephen Fitzpatrick: I didn't have much to say about this before but I can't believe what I just heard
    [P8] Stephen Fitzpatrick didn't have much to say about this before but he can't believe what he just heard
T9  Stephen Fitzpatrick: the Scottish people expect more from their politicians than the rest of the UK
    [P9] the Scottish people expect more from their politicians than the rest of the UK
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P3 Inference | - | missed Inference |
| - | P2 -> P4 Inference | invented Inference |
| - | P3 -> P4 Inference | invented Inference |
| - | P4 -> P5 Inference | invented Inference |
| P6 -> P5 Rephrase | P6 -> P5 Inference | Rephrase read as Inference |
| P7 -> P8 Inference | P7 -> P8 Inference | right Inference |
| - | P9 -> P8 Inference | invented Inference |

## nodeset20330 (13 propositions)
```text
T1  Fiona Bruce: Just to clarify, is that what you said
    [P1] just to clarify, xxx is what Angela Haggerty said
T2  Fiona Bruce: Is that what you meant
    [P2] xxx is that what Angela Haggerty meant
T3  Angela Haggerty: I'm not saying that's okay for that to be the situation
    [P3] Angela Haggerty is not saying that's okay for xxx to be the situation
T4  Angela Haggerty: we're in a crisis in an unprecedented time
    [P4] we're in a crisis in an unprecedented time
T5  Angela Haggerty: the public would be far more likely to overlook that
    [P5] the public would be far more likely to overlook that
T6  Angela Haggerty: the pandemic is what is at the forefront of everybody's mind now
    [P6] the pandemic is what is at the forefront of everybody's mind now
T7  Ian Murray: It's not about the SNPs, Labour, the Tories
    [P7] it's not about the SNPs, Labour, the Tories
T8  Ian Murray: It is about the SNP seeking the truth
    [P8] it is about the SNP seeking the truth
T9  Audience Member: Is the government doing enough to help assist the fishing and shellfish industries
    [P9] the government is or isn't doing enough to help assist the fishing and shellfish industries
T10  Michael Forsyth: The answer to that is no
    [P10] the answer to is the government doing enough to help assist the fishing and shellfish industries is no
T11  Michael Forsyth: They made some 100 million pounds available
    [P11] the goverment made some 100 million pounds available
T12  Michael Forsyth: the fact is that there is quite a lot to be sorted out in terms of the application of idiotic rules both across the straits into Europe and also within our own country in Northern Ireland
    [P12] the fact is that there is quite a lot to be sorted out in terms of the application of idiotic rules both across the straits into Europe and also within our own country in Northern Ireland
T13  Fiona Bruce: The EU say this is the deal, you have to deal with it
    [P13] the EU say this is the deal, you have to deal with it
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Rephrase | - | missed Rephrase |
| P3 -> P2 Rephrase | P3 -> P2 Rephrase | right Rephrase |
| - | P4 -> P5 Inference | invented Inference |
| P6 -> P5 Inference | P6 -> P5 Inference | right Inference |
| P8 -> P7 Inference | P8 -> P7 Rephrase | Inference read as Rephrase |
| P10 -> P9 Rephrase | P10 -> P9 Rephrase | right Rephrase |
| P11 -> P10 Inference | P11 -> P10 Conflict | Inference read as Conflict |
| P12 -> P10 Inference | P12 -> P10 Inference | right Inference |
| P13 -> P12 Conflict | - | missed Conflict |

## nodeset20331 (14 propositions)
```text
T1  Fiona Bruce: ou don't think it is divided with what is going on
    [P1] does or doesn't Jeane Freeman think the SNP is divided with what is going on
T2  Jeane Freeman: It is a big party
    [P2] the SNP is a big party
T3  Fiona Bruce: Would you describe it as united
    [P3] would or wouldn't Jeane Freeman describe the SNP as united
T4  Jeane Freeman: It has disagreements
    [P4] the SNP has disagreements
T5  Jeane Freeman: Michael belongs to a party that has disagreements
    [P5] Michael Forsyth belongs to a party that has disagreements
T6  Jeane Freeman: one currently about their Scottish leader
    [P6] one disagreement of Michael Forsyth's party is currently about their Scottish leader
T7  Jeane Freeman: Iain has had disagreements with his party
    [P7] Iain Murray has had disagreements with his party
T8  Jeane Freeman: The point is it's not uncommon for there to be disagreements between party members
    [P8] it's not uncommon for there to be disagreements between party members
T9  Jeane Freeman: that is entirely to be expected
    [P9] disagreements between party members are entirely to be expected
T10  Jeane Freeman: What isn't acceptable, and I think it is important that I say this, is any disagreements are conducted that is disrespectful of other points of view
    [P10] what isn't acceptable is any disagreements are conducted that is disrespectful of other points of view
T11  Jeane Freeman: I want to be in a political party and a country where different viewpoints and different arguments, Donald Dyer famously said, are conducted with respect and without abuse
    [P11] Jeanne Freeman wants to be in a political party and a country where different viewpoints and different arguments, Donald Dyer famously said, are conducted with respect and without abuse
T12  Fiona Bruce: Who do you think is being disrespectful then
    [P12] who does or doesn't Jeanne Freeman think is being disrespectful then
T13  Jeane Freeman: People feel that they have been -- when they have been voicing opinions on different matters that they have been not listened to
    [P13] people feel, when they have been voicing opinions on different matters, that they have been not listened to
T14  Jeane Freeman: that they have been treated disrespectfully on all sides of the different arguments and disputes going on
    [P14] people feel that they have been treated disrespectfully on all sides of the different arguments and disputes going on
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P2 -> P1 Rephrase | invented Rephrase |
| P3 -> P1 Rephrase | - | missed Rephrase |
| P4 -> P3 Inference | P4 -> P3 Rephrase | Inference read as Rephrase |
| P4 -> P6 Inference | - | missed Inference |
| - | P4 -> P8 Inference | invented Inference |
| P6 -> P5 Inference | P6 -> P5 Rephrase | Inference read as Rephrase |
| P5 -> P8 Inference | P5 -> P8 Inference | right Inference |
| P7 -> P8 Inference | P7 -> P8 Inference | right Inference |
| P9 -> P8 Rephrase | P9 -> P8 Rephrase | right Rephrase |
| P9 -> P10 Inference | - | missed Inference |
| P9 -> P11 Inference | - | missed Inference |
| P10 -> P11 Inference | P11 -> P10 Inference | Inference, wrong direction |
| P12 -> P10 Inference | - | missed Inference |
| - | P13 -> P12 Rephrase | invented Rephrase |
| P14 -> P12 Inference | - | missed Inference |
| P13 -> P14 Inference | P14 -> P13 Rephrase | Inference read as Rephrase |

## nodeset20336 (8 propositions)
```text
T1  Stephen Fitzpatrick: This is another argument in favour of the focus on rebuilding our economy
    [P1] this is another argument in favour of the focus on rebuilding our economy
T2  Stephen Fitzpatrick: there will be no support for underprivileged children and young children in coastal communities if we cannot re-start our economy
    [P2] there will be no support for underprivileged children and young children in coastal communities if we cannot re-start our economy
T3  Stephen Fitzpatrick: We have millions who have lost their jobs and savings and thousands who have lost their homes in the years to come
    [P3] we have millions who have lost their jobs and savings and thousands who have lost their homes in the years to come
T4  Stephen Fitzpatrick: We have to ask when are we getting people back to work and children back to school
    [P4] we have to ask xxx is when we are getting people back to work and children back to school
T5  Fiona Bruce: Are we not off the question here
    [P5] we are off the question here
T6  Fiona Bruce: Is the government doing enough for fishing and shellfish
    [P6] the government is or isn't doing enough for fishing and shellfish
T7  Stephen Fitzpatrick: We have economic turmoil all over the country
    [P7] we have economic turmoil all over the country
T8  Stephen Fitzpatrick: While we could have short-term support, the coastal communities need long-term support and investment
    [P8] while we could have short-term support, the coastal communities need long-term support and investment
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P2 -> P1 Inference | invented Inference |
| - | P3 -> P1 Inference | invented Inference |
| - | P4 -> P1 Inference | invented Inference |
| P3 -> P2 Inference | - | missed Inference |
| - | P5 -> P4 Conflict | invented Conflict |
| P7 -> P6 Rephrase | - | missed Rephrase |
| P8 -> P7 Inference | P7 -> P8 Inference | Inference, wrong direction |

## nodeset20338 (9 propositions)
```text
T1  Ian Murray: That's the problem
    [P1] that's the problem
T2  Ian Murray: They had to keep them in in order to stop the non-tariff barriers being put in place
    [P2] they had to keep them in in order to stop the non-tariff barriers being put in place
T3  Ian Murray: It is ridiculous
    [P3] it is ridiculous
T4  Ian Murray: The only upside of all this, and I spoke to the Scottish Fishing Federation yesterday, because the customers are depressed all over Europe but when that customer base comes back it will be dreadful unless it is done now
    [P4] the only upside of all this, Ian Murray spoke to the Scottish Fishing Federation yesterday, because the customers are depressed all over Europe but when that customer base comes back it will be dreadful unless it is done now
T5  Stephen Fitzpatrick: I grew up in a coastal town in Northern Ireland and I can tell you the problems facing coastal communities and fishing communities are not new since 1 April
    [P5] Stephen Fitzpatrick grew up in a coastal town in Northern Ireland and he can tell you the problems facing coastal communities and fishing communities are not new since 1 April
T6  Stephen Fitzpatrick: They have been undersupported for decades
    [P6] coastal fishing communities in Northern Ireland have been undersupported for decades
T7  Fiona Bruce: there is a different situation now in terms of selling their fish
    [P7] there is a different situation now in terms of selling their fish
T8  Stephen Fitzpatrick: I stand by the point the most underprivileged communities in the whole of the UK all lie along the coast
    [P8] Stephen Fitzpatrick stands by the point the most underprivileged communities in the whole of the UK all lie along the coast
T9  Stephen Fitzpatrick: They are no longer inner cities; they are young people and children in coastal communities all over the United Kingdom
    [P9] the most underprivileged communities in the whole of the uk are no longer inner cities; they are young people and children in coastal communities all over the United Kingdom
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Rephrase | P2 -> P1 Inference | Rephrase read as Inference |
| - | P2 -> P3 Inference | invented Inference |
| P6 -> P5 Inference | P6 -> P5 Inference | right Inference |
| - | P7 -> P5 Conflict | invented Conflict |
| P7 -> P6 Conflict | - | missed Conflict |
| P9 -> P8 Rephrase | P9 -> P8 Inference | Rephrase read as Inference |

## nodeset20339 (12 propositions)
```text
T1  Ian Murray: The Scottish fishing Federation chief executive said herself we must be the only coastal state that has full control over our waters but we are fishing with one hand tied behind our backs
    [P1] the Scottish fishing Federation chief executive said herself we must be the only coastal state that has full control over our waters but we are fishing with one hand tied behind our backs
T2  Ian Murray: This is the deal the prime minister put together and the deal he signed
    [P2] this is the deal the prime minister put together and the deal he signed
T3  Ian Murray: The Northern Ireland protocol is there
    [P3] the Northern Ireland protocol is there
T4  Ian Murray: we had to put a border in the Irish Sea
    [P4] we had to put a border in the Irish Sea
T5  Ian Murray: the border could not go on the island of Ireland
    [P5] the border could not go on the island of Ireland
T6  Ian Murray: This is a consequence of leaving a trading union with one of your biggest trading partners
    [P6] the protocol is a consequence of leaving a trading union with one of your biggest trading partners
T7  Ian Murray: It is not a consequence of stupid rules
    [P7] the protocol is not a consequence of stupid rules
T8  Ian Murray: it is a consequence of what has been signed
    [P8] the protocol is a consequence of what has been signed
T9  Michael Forsyth: Northern Ireland is part of the United Kingdom
    [P9] Northern Ireland is part of the United Kingdom
T10  Ian Murray: Yes
T11  Ian Murray: not in terms of the single market
    [P10] not in terms of the single market
T12  Ian Murray: The border in the Irish Sea, in the customs union
    [P11] the border in the Irish Sea, in the customs union
T13  Fiona Bruce: We're applying custom union rules
    [P12] we're applying custom union rules
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Inference | P1 -> P2 Inference | Inference, wrong direction |
| P8 -> P2 Rephrase | - | missed Rephrase |
| P4 -> P3 Inference | P4 -> P3 Inference | right Inference |
| - | P6 -> P3 Inference | invented Inference |
| P5 -> P4 Inference | P5 -> P4 Inference | right Inference |
| - | P7 -> P6 Conflict | invented Conflict |
| - | P8 -> P6 Rephrase | invented Rephrase |
| P7 -> P8 Inference | - | missed Inference |
| P10 -> P9 Conflict | P10 -> P9 Conflict | right Conflict |
| P11 -> P10 Inference | P11 -> P10 Rephrase | Inference read as Rephrase |
| - | P12 -> P11 Rephrase | invented Rephrase |
