# Pilot errors: pilot20-k4-xhigh-v1

```json
{
 "run": "pilot20-k4-xhigh-v1",
 "error_kinds": {
  "invented Inference": 52,
  "right Rephrase": 47,
  "right Inference": 29,
  "missed Inference": 24,
  "invented Rephrase": 20,
  "missed Rephrase": 14,
  "Rephrase read as Inference": 13,
  "Inference read as Rephrase": 11,
  "missed Conflict": 6,
  "invented Conflict": 5,
  "Inference, wrong direction": 4,
  "right Conflict": 3,
  "Conflict read as Inference": 2,
  "Rephrase read as Conflict": 2,
  "Conflict read as Rephrase": 1,
  "Inference read as Conflict": 1
 },
 "median_turn_distance": {
  "missed": 1.0,
  "invented": 1,
  "right": 1
 }
}
```

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
| P3 -> P2 Conflict | P3 -> P2 Conflict | right Conflict |
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| P5 -> P6 Inference | P5 -> P6 Inference | right Inference |
| - | P7 -> P6 Rephrase | invented Rephrase |
| P8 -> P6 Conflict | P8 -> P6 Conflict | right Conflict |
| P10 -> P7 Conflict | P10 -> P7 Rephrase | Conflict read as Rephrase |
| - | P9 -> P10 Inference | invented Inference |
| P12 -> P11 Rephrase | P11 -> P12 Inference | Rephrase read as Inference |

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
| P2 -> P1 Rephrase | P2 -> P1 Rephrase | right Rephrase |
| P3 -> P2 Rephrase | P3 -> P2 Rephrase | right Rephrase |
| - | P4 -> P5 Inference | invented Inference |
| P6 -> P5 Inference | P6 -> P5 Inference | right Inference |
| P8 -> P7 Inference | P8 -> P7 Rephrase | Inference read as Rephrase |
| P10 -> P9 Rephrase | P10 -> P9 Rephrase | right Rephrase |
| P11 -> P10 Inference | P11 -> P10 Conflict | Inference read as Conflict |
| P12 -> P10 Inference | P12 -> P10 Inference | right Inference |
| P13 -> P12 Conflict | P13 -> P12 Inference | Conflict read as Inference |

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
| - | P8 -> P5 Inference | invented Inference |
| P7 -> P6 Conflict | - | missed Conflict |
| P9 -> P8 Rephrase | P9 -> P8 Rephrase | right Rephrase |

## nodeset20496 (17 propositions)
```text
T1  Audience Member5: I think Grant is fine saying what he said
    [P1] Audience Member 5 thinks Grant is fine saying what he said
T2  Audience Member5: He has not really answered my question
    [P2] he has not really answered Audience Member 5's question
T3  Audience Member5: I have six seats on a kerb
    [P3] Audience Member 5 has six seats on a kerb
T4  Audience Member5: I don't have a sprawling Mediterranean terrace
    [P4] Audience Member 5 does not have a sprawling Mediterranean terrace
T5  Audience Member5: I'm sick of waiting one or two days for Richi Sunac to announce the right thing, now I have to wait a week
    [P5] Audience Member 5 is sick of waiting one or two days for Richi Sunac to announce the right thing, now Audience Member 5 has to wait a week to hear what Richi Sunac announces
T6  Audience Member5: That anxiety is unfair
    [P6] the anxiety of having to wait a week to hear what Richi Sunac announces is unfair
T7  Audience Member5: We shouldn't have to wait for a budget
    [P7] we shouldn't have to wait for a budget
T8  Audience Member5: The Panel have said things I agree with
    [P8] the Panel have said things Audience Member 5 agrees with
T9  Audience Member5: I want the correct support between now and when I'm supposed to open
    [P9] Audience Member 5 wants the correct support between now and when he's supposed to open his business
T10  Audience Member5: why do my one-off grants have to last me longer than the retail sector, yet they have opened first both times
    [P10] the reason of Audience Member 5's one-off grants have to last him longer than the retail sector, yet they have opened first both times is xxx
T11  Audience Member5: someone could chime in and say I've had a VAT cut
    [P11] someone could chime in and say Audience Member 5 has had a VAT cut
T12  Audience Member5: 15 percent VAT saving on zero turnover
    [P12] someone could chime in and say Audience Member 5 has had 15 percent VAT saving on zero turnover
T13  Audience Member5: It is zero savings
    [P13] 15 percent VAT is zero savings
T14  Audience Member5: Extend it now in good faith
    [P14] extend it now in good faith
T15  Audience Member5: Where's my job retention bonus
    [P15] Audience Member 5's job retention bonus is from xxx
T16  Audience Member5: I already spent it
    [P16] Audience Member 5 has already been spent his job retention bonus
T17  Audience Member5: I spent it paying my team
    [P17] Audience Member 5 spent his job retention bonus paying his team
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P3 -> P2 Inference | invented Inference |
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| P6 -> P5 Rephrase | P5 -> P6 Inference | Rephrase read as Inference |
| P7 -> P6 Rephrase | P6 -> P7 Inference | Rephrase read as Inference |
| P8 -> P9 Inference | - | missed Inference |
| - | P10 -> P9 Inference | invented Inference |
| P12 -> P11 Rephrase | P12 -> P11 Rephrase | right Rephrase |
| P13 -> P12 Rephrase | P13 -> P12 Conflict | Rephrase read as Conflict |
| - | P13 -> P14 Inference | invented Inference |
| P16 -> P15 Rephrase | P16 -> P15 Rephrase | right Rephrase |
| P17 -> P15 Rephrase | - | missed Rephrase |
| - | P17 -> P16 Rephrase | invented Rephrase |

## nodeset20499 (19 propositions)
```text
T1  Audience Member6: Why not vaccine passports to allow businesses to ensure COVID safety when entering indoor venues
    [P1] why not vaccine passports to allow businesses to ensure COVID safety when entering indoor venues
T2  Fiona Bruce: you run a hair dressing salon, don't you, Colette
    [P2] Audience Member 7 runs a hair dressing salon
T3  Fiona Bruce: Is that what you would like
    [P3] vaccine passports is or is not what Audience Member 7 would like
T4  Audience Member7: I represent 11 hair salons across the country
    [P4] Audience Member 7 represents 11 hair salons across the country
T5  Audience Member7: Our normal practice is we are very safe
    [P5] our normal practice is we are very safe
T6  Audience Member7: we are very clean
    [P6] we are very clean
T7  Audience Member7: It is part of our training
    [P7] it is part of our training
T8  Audience Member7: we do need something like this
    [P8] we do need something like vaccine passports
T9  Audience Member7: We're being told to have COVID safe certificates in our window
    [P9] we're being told to have COVID safe certificates in our window
T10  Audience Member7: we're spending tens of thousands on all the PPE equipment
    [P10] we're spending tens of thousands on all the PPE equipment
T11  Audience Member7: it is absolutely the right thing
    [P11] vaccine passports is absolutely the right thing
T12  Audience Member7: we can protect the interests of our employees and our clients
    [P12] we can protect the interests of our employees and our clients
T13  Audience Member7: people know they are going to a COVID-safe environment
    [P13] people know they are going to a COVID-safe environment
T14  Audience Member7: Indoor is safer than outdoor
    [P14] indoor is safer than outdoor
T15  Audience Member7: some indoor venues are on a medical basis
    [P15] some indoor venues are on a medical basis
T16  Audience Member7: You can have people sitting cheek-to-cheek on a park bench, yet are we saying hospitals are not safe
    [P16] you can have people sitting cheek-to-cheek on a park bench, yet are we saying hospitals are not safe
T17  Audience Member7: they are indoors
    [P17] hospitals are indoors
T18  Audience Member7: There's all kinds of things here
    [P18] there's all kinds of things here
T19  Audience Member7: it is the right way to go
    [P19] vaccine passports is the right way to go
```
| Gold | Predicted | Verdict |
|---|---|---|
| - | P3 -> P1 Rephrase | invented Rephrase |
| P4 -> P2 Inference | - | missed Inference |
| P8 -> P3 Rephrase | P8 -> P3 Rephrase | right Rephrase |
| - | P4 -> P8 Inference | invented Inference |
| P6 -> P5 Rephrase | P6 -> P5 Rephrase | right Rephrase |
| P7 -> P6 Inference | P7 -> P6 Inference | right Inference |
| - | P9 -> P8 Inference | invented Inference |
| - | P10 -> P8 Inference | invented Inference |
| P11 -> P8 Rephrase | P11 -> P8 Rephrase | right Rephrase |
| P9 -> P11 Inference | - | missed Inference |
| P10 -> P11 Inference | - | missed Inference |
| P11 -> P12 Inference | P12 -> P11 Inference | Inference, wrong direction |
| P11 -> P13 Inference | - | missed Inference |
| - | P14 -> P11 Inference | invented Inference |
| P19 -> P11 Rephrase | P19 -> P11 Rephrase | right Rephrase |
| - | P13 -> P12 Inference | invented Inference |
| - | P15 -> P14 Inference | invented Inference |
| - | P16 -> P15 Inference | invented Inference |
| P17 -> P16 Inference | P17 -> P16 Inference | right Inference |
| - | P18 -> P19 Inference | invented Inference |

## nodeset20505 (14 propositions)
```text
T1  Tony Danker: The third thing I think everyone is saying is this can't be the new normal
    [P1] the third thing Tony Danker thinks everyone is saying is this can't be the new normal
T2  Tony Danker: We can't build a society where everybody carries around health cards
    [P2] we can't build a society where everybody carries around health cards
T3  Tony Danker: This is different from what happened after 9/11
    [P3] this is different from what happened after 9/11
T4  Tony Danker: This would be useful and welcomed
    [P4] this would be useful and welcomed
T5  Tony Danker: it would help in the short run
    [P5] it would help in the short run
T6  Tony Danker: this is for Michael Gove in his review, you have to talk to people on the frontline to make it happen
    [P6] this is for Michael Gove in his review, you have to talk to people on the frontline to make it happen
T7  Tony Danker: We saw with supermarkets when it came to masks, mask wearing, that was incredibly difficult to operate
    [P7] we saw with supermarkets when it came to masks, mask wearing, that was incredibly difficult to operate
T8  Tony Danker: there are ethical issues, but before you get to that, let's make sure this is implementable
    [P8] there are ethical issues, but before you get to that, let's make sure mask wearing is implementable
T9  Tony Danker: Let's get talking to people on this call working in restaurants and hair salons, about how to do it
    [P9] let's get talking to people on this call working in restaurants and hair salons, about how to do implement wearing masks
T10  Tony Danker: make sure we could do it
    [P10] let's make sure we could implement wearing masks
T11  Tony Danker: it helps in the short run
    [P11] wearing masks helps in the short run
T12  Tony Danker: it is definitely work worth a look
    [P12] wearing masks is definitely work worth a look
T13  Fiona Bruce: We had a vaccines minister saying it is out the question
    [P13] we had a vaccines minister saying it is out the question
T14  Fiona Bruce: it is not out the question because the government is doing this
    [P14] it is not out the question because the government is doing this
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Inference | P2 -> P1 Rephrase | Inference read as Rephrase |
| P3 -> P1 Inference | - | missed Inference |
| - | P3 -> P2 Inference | invented Inference |
| P5 -> P4 Inference | P5 -> P4 Inference | right Inference |
| - | P11 -> P5 Rephrase | invented Rephrase |
| - | P7 -> P6 Inference | invented Inference |
| - | P8 -> P6 Rephrase | invented Rephrase |
| P8 -> P9 Inference | P9 -> P8 Rephrase | Inference read as Rephrase |
| P10 -> P8 Rephrase | - | missed Rephrase |
| - | P10 -> P9 Rephrase | invented Rephrase |
| P11 -> P12 Inference | P11 -> P12 Inference | right Inference |
| P14 -> P13 Conflict | P14 -> P13 Conflict | right Conflict |

## nodeset20506 (12 propositions)
```text
T1  Audience Member12: Matt Hancock claimed there never was a national PPE shortage during the pandemic
    [P1] Matt Hancock claimed there never was a national PPE shortage during the pandemic
T2  Audience Member12: I find this quite disrespectful to all the healthcare workers that have been fighting for their safety during the pandemic
    [P2] Audience Member12 finds claiming there never was a national PPE shortage during the pandemic quite disrespectful to all the healthcare workers that have been fighting for their safety during the pandemic
T3  Audience Member12: can the government explain why they think there were no shortages
    [P3] the government can or cannot explain why they think there were no shortages
T4  Anneliese Dodds: it has been very difficult to explain why there were no shortages
    [P4] it has been very difficult to explain why there were no shortages
T5  Anneliese Dodds: we know there were shortages
    [P5] we know there were shortages
T6  Anneliese Dodds: I know that myself from having talked to health professionals, people who were deeply frustrated, who were actually in some cases quite frightened, to be honest with you, about the working situation
    [P6] Anneliese Dodds knows that there were shortages from having talked to health professionals, people who were deeply frustrated, who were actually in some cases quite frightened about the working situation
T7  Anneliese Dodds: They were at that frontline
    [P7] health professionals were at that frontline
T8  Anneliese Dodds: it is deeply disappointing that we don't have the openness about what went wrong
    [P8] it is deeply disappointing that we don't have the openness about what went wrong
T9  Anneliese Dodds: it was Jeremy, who said we need to learn from what happened during this crisis
T10  Anneliese Dodds: We need to understand, you know, when there were those problems with supply, how we can make sure that we don't end up in that position again
    [P9] we need to understand, when there were those problems with supply, how we can make sure that we don't end up in that position again
T11  Anneliese Dodds: I think that's critically important
    [P10] Anneliese Dodds thinks that's critically important that we don't end up in that position again
T12  Anneliese Dodds: Let's learn from those at the frontline so we can ensure they have what they need into the future
    [P11] let's learn from those at the frontline so we can ensure they have what they need into the future
T13  Jeremy King: we need to learn from what happened during this crisis
    [P12] we need to learn from what happened during this crisis
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Rephrase | P1 -> P2 Inference | Rephrase read as Inference |
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| P5 -> P4 Conflict | P5 -> P4 Inference | Conflict read as Inference |
| P6 -> P5 Rephrase | P6 -> P5 Inference | Rephrase read as Inference |
| - | P7 -> P6 Inference | invented Inference |
| - | P9 -> P8 Inference | invented Inference |
| P8 -> P12 Inference | - | missed Inference |
| P10 -> P9 Rephrase | P10 -> P9 Inference | Rephrase read as Inference |
| - | P11 -> P9 Rephrase | invented Rephrase |
| P9 -> P12 Rephrase | P12 -> P9 Rephrase | right Rephrase |
| P11 -> P12 Rephrase | - | missed Rephrase |

## nodeset20517 (11 propositions)
```text
T1  Audience Member 16: Why do the government think that summer schools are a good way for students to catch up
    [P1] xxx is why the government thinks that summer schools are a good way for students to catch up
T2  Fiona Bruce: This is a new announcement about summer schools
    [P2] this is a new announcement about summer schools
T3  Fiona Bruce: Do you think it would be a good idea
    [P3] Audience Member 16 does or does not think summer schools would be a good idea
T4  Audience Member 16: the problem with it is that certainly for disadvantaged students, they are not going to take up the opportunity
    [P4] the problem with it is that certainly for disadvantaged students, they are not going to take up the opportunity
T5  Audience Member 16: The EEF did a pilot
    [P5] the EEF did a pilot
T6  Audience Member 16: about 50 percent didn't take up the opportunity
    [P6] about 50 percent didn't take up the opportunity
T7  Audience Member 16: school leaders know their cohorts, their communities
    [P7] school leaders know their cohorts, their communities
T8  Audience Member 16: they are best placed to decide how to do the catchup
    [P8] school leaders are best placed to decide how to do the catchup
T9  Fiona Bruce: You think just leave it up to the schools
    [P9] Audience Member 16 thinks or does not think to just leave the decision on the catchup to the schools
T10  Audience Member 16: Yes
T11  Audience Member 16: they should make the decisions
    [P10] schools should make the decisions on the catchup
T12  Audience Member 16: They know their community, they know their students
    [P11] schools know their community, they know their students
```
| Gold | Predicted | Verdict |
|---|---|---|
| P4 -> P3 Rephrase | P4 -> P3 Rephrase | right Rephrase |
| - | P5 -> P4 Inference | invented Inference |
| P6 -> P4 Rephrase | P6 -> P4 Inference | Rephrase read as Inference |
| P7 -> P8 Inference | P7 -> P8 Inference | right Inference |
| P11 -> P7 Rephrase | - | missed Rephrase |
| P10 -> P8 Rephrase | - | missed Rephrase |
| P10 -> P9 Rephrase | P10 -> P9 Rephrase | right Rephrase |
| P11 -> P10 Inference | P11 -> P10 Inference | right Inference |

## nodeset20518 (13 propositions)
```text
T1  Fiona Bruce: You don't think there was a shortage
    [P1] Tony Danker doesn't think there was a shortage
T2  Tony Danker: This is like stocking supermarkets across the country
    [P2] supplying PPE is like stocking supermarkets across the country
T3  Tony Danker: there was a shortage by definition because there wasn't enough in the place it needed to be all the time
    [P3] there was a shortage by definition because there wasn't enough in the place it needed to be all the time
T4  Tony Danker: We're going to need 1 million loaves of bread and we'll spread this to them
    [P4] we're going to need 1 million loaves of bread and we'll spread this to them
T5  Tony Danker: You need 2 million if there's going to be a million bought
    [P5] you need 2 million loaves if there's going to be a million bought
T6  Tony Danker: what's going on here is people are sitting centrally saying we have enough stock for what the country needs
    [P6] what's going on here is people are sitting centrally saying we have enough stock for what the country needs
T7  Tony Danker: that doesn't mean every hospital has enough supply
    [P7] enough stock for the country doesn't mean every hospital has enough supply
T8  Tony Danker: what we have now is an oversupply of vaccine
    [P8] what we have now is an oversupply of vaccine
T9  Tony Danker: We have people at the end of the day who are trying to get rid of vaccines with volunteers
    [P9] we have people at the end of the day who are trying to get rid of vaccines with volunteers
T10  Tony Danker: What we didn't have with PPE, it seems to me, is the necessary oversupply of PPE
    [P10] what we didn't have with PPE, it seems to Tony Danker, is the necessary oversupply of PPE
T11  Tony Danker: it is disingenuous to say mathematically that we have enough
    [P11] it is disingenuous to say mathematically that we have enough
T12  Tony Danker: If you haven't enough in every part of the country, you haven't got enough
    [P12] if you haven't enough in every part of the country, you haven't got enough
T13  Tony Danker: this might be a maths story rather than anything too much more sinister than that
    [P13] this might be a maths story rather than anything too much more sinister than that
```
| Gold | Predicted | Verdict |
|---|---|---|
| P3 -> P1 Rephrase | P3 -> P1 Conflict | Rephrase read as Conflict |
| P2 -> P3 Rephrase | P2 -> P3 Inference | Rephrase read as Inference |
| P4 -> P2 Inference | P4 -> P2 Rephrase | Inference read as Rephrase |
| P5 -> P2 Inference | - | missed Inference |
| P6 -> P3 Inference | - | missed Inference |
| P7 -> P3 Inference | P7 -> P3 Inference | right Inference |
| - | P5 -> P4 Rephrase | invented Rephrase |
| P7 -> P6 Rephrase | - | missed Rephrase |
| P8 -> P9 Inference | P9 -> P8 Inference | Inference, wrong direction |
| - | P8 -> P10 Inference | invented Inference |
| P12 -> P11 Inference | P12 -> P11 Inference | right Inference |
| - | P11 -> P13 Inference | invented Inference |

## nodeset20521 (16 propositions)
```text
T1  Fiona Bruce: Narcissistic denial, which is is new one on Question Time
    [P1] narcissistic denial, which is new one on Question Time
T2  Fiona Bruce: Disingenuous, insulting, just not true
    [P2] disingenuous, insulting, just not true
T3  Grant Shapps: that was to math
    [P3] that was to math
T4  Fiona Bruce: You are here for the government
    [P4] Grant Shapps is here for the government
T5  Fiona Bruce: You've copped it
    [P5] Grant Shapps has copped it
T6  Grant Shapps: We're starting from a position where we delivered 32 billion pieces of PPE
    [P6] we're starting from a position where we delivered 32 billion pieces of PPE
T7  Grant Shapps: We're at the other end of the crisis
    [P7] we're at the other end of the crisis
T8  Grant Shapps: A year ago, if we had been sitting here, just beginning to hear about the coronavirus thing
    [P8] a year ago, if we had been sitting here, just beginning to hear about the coronavirus thing
T9  Grant Shapps: for years this country had not invested in doing testing
    [P9] for years this country had not invested in doing testing
T10  Grant Shapps: we couldn't test for this
    [P10] we couldn't test for coronavirus
T11  Grant Shapps: we hadn't invested in making PPE
    [P11] we hadn't invested in making PPE
T12  Grant Shapps: 1 percent of our PPE was being made in this country
    [P12] 1 percent of our PPE was being made in this country
T13  Grant Shapps: I don't want to sit here and say, oh it was all easy
    [P13] Grant Shapps doesn't want to sit here and say, oh it was all easy
T14  Grant Shapps: day in, day out, we were battling to make sure there was enough PPE in the hospitals and in the care homes, who we never had to supply before
    [P14] day in, day out, we were battling to make sure there was enough PPE in the hospitals and in the care homes, who we never had to supply before
T15  Grant Shapps: we were in that position worldwide, globally, with every other country
    [P15] we were in the position with no enough PPE worldwide, globally, with every other country
T16  Grant Shapps: the audience and people listening are right to say why didn't Britain work this out before the
    [P16] the audience and people listening are right to say why didn't Britain work this out before the
```
| Gold | Predicted | Verdict |
|---|---|---|
| P2 -> P1 Rephrase | P2 -> P1 Rephrase | right Rephrase |
| - | P4 -> P5 Inference | invented Inference |
| P6 -> P7 Inference | P6 -> P7 Inference | right Inference |
| P9 -> P10 Inference | P9 -> P10 Inference | right Inference |
| P11 -> P12 Inference | P12 -> P11 Inference | Inference, wrong direction |
| - | P11 -> P16 Inference | invented Inference |
| P14 -> P13 Inference | P14 -> P13 Inference | right Inference |
| P15 -> P14 Rephrase | - | missed Rephrase |
| - | P15 -> P16 Inference | invented Inference |
