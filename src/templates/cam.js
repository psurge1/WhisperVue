import '../App.css';


const VidStream = ({stream, altLabel}) => {
    return (
      <div className="stream">
        <img src={stream} alt={altLabel}/>
      </div>
    )
};

export default VidStream;