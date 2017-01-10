"""
Code to extract the occupations from the osha file
"""

import nltk
import csv
from nltk import word_tokenize
from nltk import pos_tag
from nltk.chunk import *
from nltk.tokenize import sent_tokenize

#s = "At approximately 11:30 a.m. on November 13  2013  Employee #1  with Edco Waste  & Recycling Services  was operating a forklift (Linde Lift Truck; Serial  Number: H2X393S04578; identified by the employer as FL-3) from approximately  4:00 a.m.  moving bales of recyclable paper products from a collection area in  the yard into trucks. Then  Employee #1 cleaned and was replacing an air  filter on the forklift FL-3. To clean out the air filter  Employee #1 parked  FL-3 in the doorway of the maintenance building. The air filter was located on  the rear of the forklift  behind the cab frame on the driver's side. Employee  #1 removed the air filter and cleaned it out  and then he climbed up onto the  back of the forklift to replace it. While up on the back of the forklift   Employee #1's foot dislodged the cooling system radiator cap. The fluid in the  lift truck's cooling system was hot and under pressure from being operated all  morning. The hot fluid sprayed up and out of the reservoir. Employee #1 was  burned on the upper legs and the groin area. Employee #1 jumped off of the  back of the forklift onto the ground. Coworkers came to his assistance and  called emergency services. Employee #1 was hospitalized at a burn center for  over 24 hours  for treatment of second degree burns to the upper legs and  groin area."
##s = "The employee  an electrician  was removing unused and deenergized light  fixtures  wiring  and piping  from a ceiling directly above open   solution-filled tanks of an automatic plating line. He was standing on the  open structural steel support frame for the automatic overhead crane tracks.  The employee apparently lost his balance  falling approximately 9 ft  striking  his head and falling into the rinse tank containing ammonium chloride and zinc  chloride. He was removed from the tank by fellow employees and taken to the  hospital where he died approximately two hours later. The cause of death was  listed as severe adult respiratory distress syndrome  caused by chemical burns  to the lungs. At the time of the accident  the deceased was not tied off  nor  was the structural steel provided with guardrails."
#s = "ON OCTOBER  31 1985  AT APPROXIMATELY 12 AM  EMPLOYEE #1 COLLAPSED JUST AFTER  TAKING A SHORT BREAK FROM A PROCESS REFERRED TO AS STRIPPING. PARTS ARE  DIPPED INTO A SERIES OF TANKS WHICH CONTAIN WATER  WATER MIXED WITH DETERGENT  AND WATER MIXED WITH HYDROCHLORIC ACID. THE ACID REMOVES CORROSION AND DIRT  FROM THE PARTS. AFTER PARTS WERE STRIPPED THEY WERE ZINC PLATED IN ANOTHER  PROCESS OPERATED BY CO-WORKERS. EMPLOYEE #1 WAS NOT WORKING WITH ANY CYANIDE  OR CYANIDE SOLUTIONS AT THE TIME OF THE INCIDENT OR PRIOR TO THE INCIDENT.  CO-WORKERS (3) HAD JUST FINISHED TALKING TO THE EMPLOYEE PRIOR TO HIS  COLLAPSE. HE HAD NOT COMPLAINED OF ANY ILL FEELINGS AND CO-WORKERS HAD NOT  EXPERIENCED ANY ADVERSE AFFECTS. NO EXPLANATION COULD BE OFFERED BY  CO-WORKERS. THE EMPLOYEE DIED ABOUT 2 DAYS AFTER HE COLLAPSED. ATTENDING  PHYSICIANS DIAGNOSED HIS CONDITION AS ACUTE CYANIDE POISONING."
#s = "InspectionOpen DateSICEstablishment Name"
#s = "On April 16  2003  Employee #1  a laborer  was operating a tractor pulling a  hay cutter that was engaged by the powered-take-off (PTO) on the tractor. The  employee got off of the tractor to disengage the PTO by pulling on a handle  behind the tractor seat. When he pulled on the handle  it broke off and his  arm contacted the unguarded PTO which amputated his arm. He was taken to Kern  Medical Center for emergency treatment. On May 26  2003  he died from  complications."
#s = "On July 30  2003  Employee #1 was a crew member working on dismantling a post  and a beam farm barn. He was removing the beams that supported the roof  rafters using a JCB 805C rough terrain fork truck. A 20-ft long  8  in.-by-8-in. beam  attached to the forks by fabric straps  was being lifted by  the truck. It started to descend when the truck tipped to the left. The truck  rolled over on its left side bringing to the ground the forks and Employee #1  who was working from the truck forks. Employee #1 was killed in an  approximately 30-ft fall from a crane. The autopsy report listed traumatic  injuries  including large laceration of the right ventricle  as the cause of  death."
#s = " At approximately 12:08 p.m. on June 16  2010  Employee #1  a window cleaning  operator  and a coworker were preparing to start window cleaning operations of  an 18-story residential building. Employee #1 was sitting on a parapet  which  had a smooth sheet metal cover  of the building. Employee #1 was not attached  to the rope descent system. Employee #1 disconnected his lanyard from the life  line to reposition the life line to his other side. While turning to move the  life line  Employee #1 slipped and fell over the edge of the building.  Employee #1 held on to the rope until he reached the ground. Employee #1  sustained degloving injuries to both hands  a contusion and abrasion to his  left elbow  and abrasions to the right side of his face. Employee #1 was  hospitalized."
###s = "At 4:00 p.m. on July 20  2012  an employee was working as a member of a  sanitation crew at Ramco Enterprises  L.P.  in Moss Landing  CA. He was not a  contract employee. He was using a hose to clean a conveyor belt. The machinery  was guarded  but the employee removed the guard to do a better job of cleaning  the apparatus. His right hand became stuck in a chain  and he sustained a  partial amputation of a fingertip. The employee was taken to Watsonville  Community Hospital  where he was treated and released after surgery. The  accident was reported by the firm's safety compliance manager at 5:59 p.m.  that same day.                                                                  "
###s = "On June 21  2007  at approximately 10:00 a.m.  Employee #1  the hooktender   was finishing the logging operations on a yarder skid road. He had hooked a  turn of logs to send to the landing. One of the logs hung up and swung out   striking Employee #1 in the back. Employee #1 was hospitalized for treatment  of his injuries.                           "
#s = "On July 14  2013  Employee #1  vacuum pump truck driver and operator  was  offloading hot brine water at a geothermal power plant. He was assigned to  transfer loads of the brine between power plants and had already made several  trips between facilities. When he arrived at the plant  Employee #1 connected  the hose to the vehicle's tank outlet valve and proceeded to empty the tanks  contents by gravity. While the tank was left emptying  he went inside the  plant's control center briefly to cool off and to get a drink of water.  Employee #1 noticed that the flow of brine had stopped due to a clog in the  hose when he returned. He tried to clear the hose by switching the truck to  vacuum. As he did this  he noticed that the flow was still impeded. Employee  #1 loosened the hose coupler on the truck's valve  which caused hot  pressurized brine to flow out. He attempted to hold the hose  but eventually  let go  causing the hot brine water to splash on his left abdomen  right leg   left leg and left shoulder. Employee #1 rushed into the control room and  removed his hot brine soaked clothing. A coworker observed the pump truck with  the brine pouring out and Employee #1 running toward the control room. This  coworker shut off the valve on the pump truck and went to check on Employee  #1. Employee #1 was taken to a hospital and was then transferred to the burn  unit of a medical center. He was admitted to the medical center  where he was  treated for second and third-degree burns and then hospitalized.                "
#s = " At approximately 6:30 a.m. on May 13  2013  Employee #1  a foreman regularly  employed by Integrity Rebar Placers  was operating a rough terrain forklift at  a contracted job site in Murrieta  CA. He was using the forklift to move  bundles of steel. Employee #1 was positioning the forklift to pick up another  load when the forklift tipped back and over. Employee #1 exited the cab of the  forklift as the machine tipped over. The forklift fell on top of Employee #1   pinning him under the lower section of the boom and crushing his abdomen. He  was killed. The employer notified Cal/OSHA of this fatality at approximately  8:35 a.m. on May 13  2013. The subsequent investigation determined that  Employee #1 had been employed by the company for approximately 2.5 months.      "
#s = " On April 23  2013  Employee #1  a baggage tug operator  was working on the  ground. A coworker was pulling a cart with a tug making turns toward the exit  gate. The turning cart and tug struck and ran over the employee. Employee #1  was transported to King County Medical Examiners  where he was treated for an  abdominal fracture. Employee #1 remained hospitalized.                          "
#s = "At approximately 8:00 am  on May 3  2010  Employee #1  a maintenance mechanic  of Jim's Supply Co  Inc.  sustained a leg injury while in the course of his  normal work duties. Employee #1 was in the process of replacing a hydraulic  pressure valve from a roll form machine. Employee #1 was injured when he  removed the valve while the system was still under pressure. As he used a pipe  wrench and a cheater bar to turn the valve  the valve came loose and struck  him on the leg. Employee #1 received a right femur shaft open fracture and was  transported to Kern Medical Center for treatment of his fractures.              "
#s = "At 9:15 a.m. on May 11  2011  Employee #1  a power-plant operator  was working  for an employer that specialized in the assembly of batteries. As Employee #1  was working in the battery supply assembly area  using a Dork pneumatic air  wrench tool to remove bolts from a battery frame  his left glove was caught by  the rotator tip of the wrench he was using. This caused his wrist to sprain  and his left thumb was amputated. Employee #1 was initially taken to U.S.  Health Work Clinic  where he was evaluated and then transferred to Western  Medical Center in Santa Ana  California. There he was treated for his  injuries. The accident investigation determined that no standard  rule  order  or regulation  set forth in Title 8 of the California Code of Regulations  was  violated in connection with the accident.                                       "
#s = "On April 19  2012  Employee #1  a psychiatric nurse with Orange County  Industrial Plastics  Western Medical Center  was caring for a patient who had  been admitted for behaving in an agitated manner. Employee #1  who was  momentarily distracted and looking down at some paperwork  was approached by  the patient and struck violently in the head. Employee #1 fell off of his  chair and struck his head on the floor. Other employees came to his assistance  and restrained the patient. Employee #1 was hospitalized and underwent surgery  for a fractured skull.                                                          "
#s = "At 7:07 p.m. on May 8  2012  Employee #1  a psychiatric technician  was  attacked by a patient at the Napa State Hospital. Employee #1 was hospitalized  for over 24 hours for treatment of a fractured ankle  facial laceration and a  shoulder injury.           "
#s = "On August 7  2008  a worker of Jerry Nolan  Inc. d/b/a Creative Sign Company  was in an aerial basket  being elevated  so that he could replace a sign at a  local fast food restaurant. The crane was located in the parking lot of an  adjacent seafood restaurant  and positioned next to energized overhead power  lines. As work progressed  the worker contacted an energized 7.6-kV overhead  power line. The employer did not deenergize the overhead lines before work  began. An investigation revealed that a safe minimal clearance between the  overhead power lines and crane was not maintained. Following the incident  the  distance measured between the workman's basket and the overhead power lines  was approximately 3 ft. At the time of the accident  the operator's controls  in the workman's basket were inoperable. A coworker was elevating the  workman's basket using the base control box. It was also revealed that the  employer did not develop and implement safe work practices for work performed  near or around overhead power lines.          "
#s = "Sometime before 2:00 p.m. on July 20  2012  Employee #1 was working as a  painter in Santa Clara CA. He had two employers. Barrett Business Services   Inc.  was his primary employer  and Ekim Painting - North  Inc.  was his  secondary employer. Based on a written contract between Ekim Painting - North   Inc.  and Barrett Business Services  Inc.  the secondary employer was  responsible for Employee #1's work safety and training. Employee #1 stated  that  on the day of the incident  he had been assigned by his supervisor to  paint roof gutters on an exterior part of a two-story residential house.  Employee #1 was up on the roof. He stated that he was standing on a 6-foot  (1.8-meter) stepladder. It had been placed on the roof. The roof had a  slightly sloped and uneven surface  as it was covered with curved ceramic  tiles. Employee #1 stated that he was using this stepladder as a single  ladder. That is  he was using it without opening it. Rather than have all four  of the ladder's feet on the surface of the roof  he had only two of the  ladder's feet on the roof. The top of the ladder was leaning against a wall.  Employee #1 stated that the ladder was accordingly not secure. While Employee  #1 was on the ladder  it started slipping off the roof. Employee #1 fell off  the roof. As he fell  his body struck a stucco column with a 2-foot by 2-foot  (0.6-meter by 0.6-meter) surface  which was about 10 feet (3 meters) below the  roof. He then bounced to the stairways below the stucco column  where he hit  his head on the metal railing of stairways. He landed on the sidewalk and  finally stopped falling. The total fall distance from the roof to the ground  was about 19 feet (5.8 meters). Employee #1 suffered a laceration to his head  and abrasions to his left side of the body. The Santa Clara  CA  fire  department responded. Employee #1 was hospitalized for this injury  but not  for longer than 24 hours. A coworker witnessed this accident directly while he  was working near Employee #1. At about 2:00 p.m. on July 20  2012  The Santa  Clara fire department reported the injury to the San Mateo District office.  Both of Employee #1's employers reported this incident in a timely manner as  well. An inspection was initiated on August 18  2012. Both Employee #1 and the  coworker stated in interviews that it was a common practice at the jobsite to  use a stepladder on a sloped surface as a single ladder. During the  investigation  a report from the Santa Clara fire department  which included  pictures of the accident scene  was reviewed. Based on statements given  the  employer (the narrative did not say which one) was cited for two serious  violations. The employer failed to ensure that the stepladder was placed on  secure and level footings (T8 CCR 3276 (e) (7). The employer failed to ensure  that the stepladder was not used as a single ladder (T8 CCR 3276 (e)(16)(C).    "
#s = " At approximately  3:30 p.m. on July 27  2012  Employee #1 was working as a  painter for Jim Zhang  dba Yong Nian Zhang  in San Francisco  CA. The company  was a painting contractor  and it was painting the north stairwell on the  north wall of a building. Employee #1 climbed over a parapet wall to access a  two-point suspended scaffold. He was not tied off. He fell three stories from  the fourth floor down to the second floor rooftop. He sustained serious  injuries  the nature and locations of which were not given by the narrative.  He was transported to San Francisco General Hospital. The employee was later  transferred to a rehabilitation center in San Francisco  where he was  eventually released. A serious accident-related violation was cited for 8CCR  1658(m)  as well as two serious violations for 8CCR 1658(r) and 8CCR 1660(d).  Other violations were also noted and the employer was cited accordingly. The  San Francisco fire department reported the incident to the Division at 3:55  p.m. that same day. The owner of the company was present during an inspection   but he could not speak English. His son  however  could speak English  and he  translated for the two inspectors. The owner's son stated that the two-point  suspended scaffold was his father's  that the employee climbed over the  parapet wall to access the scaffold  and that the employee was not tied off.    "
#s = "On November 12  2011  Employee #1  a cook for Molly B's Family Restaurant  was  off duty  but was at the restaurant. For some reason  Employee #1 was on a  nonspecified platform and fell off  which resulted in his death. There were no  other details given in the abstract.      "
#s = "At approximately 10:45 a.m. on March 25  2003  an employee who was performing  flagman duties was struck by a 1994 Red Ford Taurus that was traveling  southeast on Walnut Grove Road. The flagman was wearing a hard hat and holding  a STOP/SLOW sign. However  the flagman did not have on high visibility  clothing (reflective vest) nor were highway (orange) cones strategically  placed throughout the work-zone. The employee was pronounced dead at the  scene.                                                                          "
#s = "On November 27  2012  Employee #1  a Champlain EMS volunteer was operating an  ambulance when the vehicle went off the road. Employee #1  the driver of the  ambulance was pronounced dead at the scene  and three others involved in the  event were transported to the Plattsburgh Hospital.                  "
#s = "On October 2  2012  Employee #1  a 68-year-old male log truck driver with  Pisgah Hardwood Corporation  was leaving a logging site with a loaded truck.  Employee #1 lost control of the vehicle and was not able to gain control of  his truck once it started rolling downhill. Employee #1jumped from truck's cab  just before the truck crashed into a large tree. Employee #1 was hospitalized  and suffered from unspecified lacerations from the event.     "
#s = "On October 25  2005  Employee #1 was a temporary worker employed by A Dream  Team Staffing. He was assigned to Medline Industries  a medical supply  wholesaler  as a laborer. Medline trained him to use a Towmotor forklift truck  or one similar to it. Employee #1  however  decided on his own to use another  forklift  one for which he had not been trained. In some manner  he crushed  his left foot and fractured it. He was hospitalized. The only equipment  involved was the forklift.                                                      "
#s = "At approximately 10:45 a.m. on March 25  2003  an employee who was performing  flagman duties was struck by a 1994 Red Ford Taurus that was traveling  southeast on Walnut Grove Road. The flagman was wearing a hard hat and holding  a STOP/SLOW sign. However  the flagman did not have on high visibility  clothing (reflective vest) nor were highway (orange) cones strategically  placed throughout the work-zone. The employee was pronounced dead at the  scene.                                                                          "
#s = "At approximatelyv11:10 a.m. on November 30  2012  Employee #1  a plumber's  assistant  was working for DNC Plumbing dba Hydrotech. The employee was using  a right-angle drill to make a vertical hole in a block of wood in order to fit  a pipe through the wood. Employee#1's right glove got caught in the drill   resulting in his finger amputation. The employee was transported to USC  Medical Center  where he was treated and released.                              " 

#occupation_grammar = r"""
#  Occ: {<NN><VBZ><VBN><IN>} 
#"""

# Chunking rules
occupation_grammar = r"""
  Occ: {<DT><NN><VBD><VBG>}
       {<VBD><DT><JJ>?<NN><VBG>}
       {<DT><NN>+<CC>}
       {<NN>+<CC><NN><VBD><VBG>}
       {<CD><DT><NN><RB>}
       {<CD><DT><JJ>?<NN>+<VBD>}
       {<CD><DT><JJ>?<NN>+<IN>}
       {<VBG><IN><DT><NN><IN>}
       {<NNP><NN><VBD>}
       {<JJ><NN>+<IN>}
       {<DT><JJ>?<NN><VBN>}
"""

#f_word = "field technician"
#index = 0
#list = f_word.split(' ')
#if 'operator' in list:
#    index = list.index('operator')
#elif  'laborer' in list:
#    index = list.index('laborer')
#elif  'driver' in list:
#    index = list.index('driver')
#elif  'mechanic' in list:
#    index = list.index('mechanic')
#elif  'technician' in list:
#    index = list.index('technician')    
#if(index>0):
#    f_word = list[index-1] + "_" + list[index] 
#print f_word

import pandas as pd
osha_file = pd.read_csv("../Datasets/Osha_Inves_Summaries.csv")

descriptions = osha_file.Description
final_list = []

for s in descriptions:
    s_list = sent_tokenize(s)
    s = s_list[0]
    
    pos = pos_tag(word_tokenize(s)) 
    #print pos
    
    test_chunker = nltk.RegexpParser(occupation_grammar)
    test_chunker_result = test_chunker.parse(pos)
    #print test_chunker_result
    
    #test_chunker_result.draw()
    tot_word = []
    tree = test_chunker_result
    for a in tree:
        if type(a) is nltk.Tree and a.label() == 'Occ':
            #print a.leaves()[0]
            for seg in a.leaves():
                if seg[1] == 'NN':
                    tot_word.append(seg[0])
    f_word = " ".join(x for x in tot_word)

    index = 0
    list = f_word.split(' ')
    if 'operator' in list:
        index = list.index('operator')
    elif  'laborer' in list:
        index = list.index('laborer')
    elif  'driver' in list:
        index = list.index('driver')
    elif  'mechanic' in list:
        index = list.index('mechanic')
    elif  'technician' in list:
        index = list.index('technician')
    elif  'technician' in list:
        index = list.index('technician')
    
    if(index>0):
        f_word = list[index-1] + "_" + list[index] 
    print f_word    
        
    final_list.append(f_word)

# Write the extracted results to a file
with open("../Datasets/OccupationsExtracted_Method2.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(final_list,descriptions))                
